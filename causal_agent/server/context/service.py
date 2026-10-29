"""Atomic context publication and independent reads. No CSV or legacy memory access."""

import hashlib

from sqlalchemy import func, select

from causal_agent.common import llm, trace
from causal_agent.memory.context import extract as E
from causal_agent.memory.context.contracts import ColumnIdentity
from causal_agent.memory.context.review import Knowledge, Revision, extracted
from causal_agent.memory.context.views import ContextColumnSummary, ContextColumnView, ContextDatasetView, ContextSummary, ContextView
from causal_agent.server.context import schema as S
from causal_agent.server.ingestion import schema as I
from causal_agent.server.ingestion.lookup import owned
from causal_agent.server.ingestion.service import Missing, uid
from causal_agent.server.profiling import schema as P


class Prerequisite(ValueError):
    pass


def source(db, owner, version_id):
    row = db.execute(select(I.contexts).where(I.contexts.c.owner_id == owner, I.contexts.c.version_id == version_id)).mappings().first()
    return (row["id"], row["text"]) if row else (None, "")


def column_index(db, owner, version_id):
    rows = db.execute(select(P.columns).where(P.columns.c.owner_id == owner, P.columns.c.version_id == version_id).order_by(P.columns.c.position)).mappings()
    return [ColumnIdentity(column_id=r["id"], position=r["position"], name=r["name"], key=r["key"]) for r in rows]


def index_ready(db, owner, info, columns):
    # A saved complete index is sufficient even when numerical checks were partial.
    counts = db.execute(
        select(P.profiles.c.dataset["columns"].as_integer()).where(
            P.profiles.c.owner_id == owner,
            P.profiles.c.version_id == info["version_id"],
            P.profiles.c.source_checksum == info["sha256"],
            P.profiles.c.status.in_(["complete", "partial"]),
        )
    ).scalars()
    if not columns or len(columns) not in counts or [c.position for c in columns] != list(range(len(columns))):
        raise Prerequisite("Profile the dataset first to establish all column identities.")


def revision_row(db, owner, version_id, revision_id=None):
    query = select(S.revisions).where(S.revisions.c.owner_id == owner, S.revisions.c.version_id == version_id)
    if revision_id is not None:
        query = query.where(S.revisions.c.id == revision_id)
    return db.execute(query.order_by(S.revisions.c.number.desc()).limit(1)).mappings().first()


def summary(db, row, dataset_id):
    knowledge = Knowledge.model_validate(row["knowledge"])
    provenance = (
        db.execute(
            select(S.packs).where(S.packs.c.id == row["source_pack_id"], S.packs.c.owner_id == row["owner_id"], S.packs.c.version_id == row["version_id"])
        )
        .mappings()
        .one()
    )
    _, text = source(db, row["owner_id"], row["version_id"])
    current = [s for s in knowledge.statements if s.status != "withdrawn"]
    return ContextSummary(
        **{
            k: provenance[k]
            for k in ("source_id", "source_checksum", "csv_checksum", "column_index_fingerprint", "schema_version", "extractor_version", "model")
        },
        revision_id=row["id"],
        version_id=row["version_id"],
        revision=row["number"],
        created_at=row["created_at"],
        dataset_id=dataset_id,
        source_present=bool(text.strip()),
        columns=[
            ContextColumnSummary(
                **c.model_dump(),
                statement_count=sum(s.scope == "column" and c.column_id in s.column_ids for s in current),
                relationship_count=sum(s.scope == "relationship" and any(p.kind == "column" and p.id == c.column_id for p in s.participants) for s in current),
            )
            for c in knowledge.columns
        ],
    )


def view(storage, owner, dataset_id):
    info = owned(storage, owner, dataset_id)
    with storage.engine.connect() as db:
        row = revision_row(db, owner, info["version_id"])
        return ContextView(status="prepared", pack=summary(db, row, dataset_id)) if row else ContextView(status="not_prepared")


def dataset(storage, owner, dataset_id, revision_id):
    info = owned(storage, owner, dataset_id)
    with storage.engine.connect() as db:
        row = revision_row(db, owner, info["version_id"], revision_id)
        if row is None:
            raise Missing("Dataset not found.")
        _, text = source(db, owner, info["version_id"])
        statements = [s for s in Knowledge.model_validate(row["knowledge"]).statements if s.status != "withdrawn"]
        return ContextDatasetView(
            pack=summary(db, row, dataset_id),
            source_text=text,
            dataset=[s for s in statements if s.scope in {"dataset", "requirement"}],
            concepts=[s for s in statements if s.scope == "concept"],
            relationships=[s for s in statements if s.scope == "relationship"],
        )


def column(storage, owner, dataset_id, revision_id, column_id):
    info = owned(storage, owner, dataset_id)
    with storage.engine.connect() as db:
        row = revision_row(db, owner, info["version_id"], revision_id)
        if row is None:
            raise Missing("Dataset not found.")
        knowledge = Knowledge.model_validate(row["knowledge"])
        identity = next((c for c in knowledge.columns if c.column_id == column_id), None)
        if identity is None:
            raise Missing("Dataset not found.")
        statements = knowledge.for_column(column_id)
        relationships = [s for s in statements if s.scope == "relationship"]
        concepts = {p.id for r in relationships for p in r.participants if p.kind != "column"}
        return ContextColumnView(
            revision_id=revision_id,
            version_id=info["version_id"],
            column=identity,
            statements=[s for s in statements if s.scope == "column"],
            dataset=[s for s in statements if s.scope in {"dataset", "requirement"}],
            relationships=relationships,
            concepts=[s for s in knowledge.statements if s.id in concepts and s.status != "withdrawn"],
        )


def publish(db, owner, info, source_id, text, columns, provenance, result):
    errors = E.validate(result.content, text, columns)
    if errors or result.assessment.corrections:
        raise E.InvalidExtraction("Publication rejected: " + "; ".join(errors + [c.model_dump_json() for c in result.assessment.corrections]))
    db.execute(select(I.versions.c.id).where(I.versions.c.id == info["version_id"], I.versions.c.owner_id == owner).with_for_update()).one()
    if revision_row(db, owner, info["version_id"]) is not None:
        raise ValueError("Context already exists. Update its current revision rather than extracting a second context.")
    pack_id = uid()
    revision = (db.scalar(select(func.max(S.packs.c.revision)).where(S.packs.c.version_id == info["version_id"])) or 0) + 1
    db.execute(
        S.packs.insert().values(
            id=pack_id,
            owner_id=owner,
            version_id=info["version_id"],
            source_id=source_id,
            revision=revision,
            **provenance,
            dataset=[s.model_dump() for s in result.content.dataset],
            concepts=[c.model_dump() for c in result.content.concepts],
            assessment=result.assessment.model_dump(),
            model_calls=result.model_calls,
        )
    )
    for col in result.content.columns:
        db.execute(
            S.column_contexts.insert().values(
                pack_id=pack_id, column_id=col.column_id, owner_id=owner, version_id=info["version_id"], statements=[s.model_dump() for s in col.statements]
            )
        )
    for relation in result.content.relationships:
        db.execute(
            S.relationships.insert().values(pack_id=pack_id, id=relation.id, owner_id=owner, version_id=info["version_id"], result=relation.model_dump())
        )
    # The extraction response is retained only as immutable provenance. The first
    # canonical revision is published in this same transaction, before any review.
    first = Revision(
        id=uid(), version_id=info["version_id"], source_pack_id=pack_id, parent_id=None, number=1, knowledge=extracted(result.content, columns, pack_id)
    )
    db.execute(S.revisions.insert().values(owner_id=owner, **first.model_dump(mode="json")))
    return first.id


def generate(storage, owner, dataset_id):
    info = owned(storage, owner, dataset_id)
    with trace.span("context:prepare", inputs={"dataset_id": dataset_id, "version_id": info["version_id"]}) as span:
        with storage.locked(owner, "context:" + info["version_id"]) as db:
            current = revision_row(db, owner, info["version_id"])
            if current is not None:
                answer = ContextView(status="prepared", pack=summary(db, current, dataset_id))
                span.end(outputs=answer.model_dump(mode="json"))
                return answer
            source_id, text = source(db, owner, info["version_id"])
            columns = column_index(db, owner, info["version_id"])
            index_ready(db, owner, info, columns)
            provenance = dict(
                source_checksum=hashlib.sha256(text.encode()).hexdigest(),
                csv_checksum=info["sha256"],
                column_index_fingerprint=E.fingerprint([c.model_dump() for c in columns]),
                schema_version=E.SCHEMA_VERSION,
                extractor_version=E.EXTRACTOR_VERSION,
                model=llm.model_name() if text.strip() else "none",
            )
            identity = E.fingerprint({**provenance, "prompts": [E.SYSTEM, E.ASSESS]})
            result = E.extract(text, columns)
            with trace.span("context:publish", inputs={"version_id": info["version_id"]}) as publication:
                revision_id = publish(db, owner, info, source_id, text, columns, {**provenance, "identity": identity}, result)
                db.commit()
                publication.end(outputs={"revision_id": revision_id})
            answer = ContextView(status="prepared", pack=summary(db, revision_row(db, owner, info["version_id"], revision_id), dataset_id))
            span.end(outputs=answer.model_dump(mode="json"))
            return answer
