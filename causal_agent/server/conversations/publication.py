"""The single publication catalogue: PostgreSQL records and checksum-verified private files."""

import hashlib
import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert

from causal_agent.common import trace
from causal_agent.common.numbers import grounded, numbers_in
from causal_agent.explore.inputs import digest, safe_file
from causal_agent.memory.artifact_tables import rows as table_rows
from causal_agent.memory.artifacts import Artifact, ArtifactElement
from causal_agent.server.conversations import schema as S
from causal_agent.server.ingestion.service import Missing, uid


class Repository:
    def __init__(self, storage, owner, conversation):
        self.storage, self.owner, self.conversation = storage, owner, conversation
        with storage.engine.connect() as db:
            self.chat = db.execute(select(S.conversations).where(S.conversations.c.id == conversation, S.conversations.c.owner_id == owner)).mappings().first()
        if self.chat is None:
            raise Missing("Conversation not found.")
        self.info = storage.get(owner, self.chat["dataset_id"])
        self.context = self.chat["context"]
        self.operation_id = None

    def begin(self, dataset, question, context, checksum, refs):
        if dataset != self.conversation or checksum != self.info["sha256"]:
            raise ValueError("Exploration must use the complete selected dataset version")
        for ref in refs:
            if ref.id.startswith("artifact:"):
                aid, name = ref.id.removeprefix("artifact:").split("/", 1)
                a = self.load(dataset, aid)
                with self.storage.engine.connect() as db:
                    saved = db.scalar(select(S.files.c.sha256).where(S.files.c.artifact_id == a.id, S.files.c.owner_id == self.owner, S.files.c.name == name))
                if saved != ref.sha256:
                    raise ValueError("Supplied artifact bytes differ from the published file")
        request = str(uuid5(NAMESPACE_URL, self.operation_id + ":exploration-request")) if self.operation_id else uid()
        attempt = str(uuid5(NAMESPACE_URL, self.operation_id + ":exploration-attempt")) if self.operation_id else uid()
        with self.storage.engine.begin() as db:
            existing = db.execute(select(S.requests).where(S.requests.c.id == request)).mappings().first()
            if existing is not None:
                expected = (
                    self.owner,
                    self.conversation,
                    self.chat["version_id"],
                    checksum,
                    question,
                    context,
                    [r.model_dump(mode="json", exclude={"path"}) for r in refs],
                )
                actual = tuple(existing[k] for k in ("owner_id", "conversation_id", "version_id", "source_sha256", "question", "context", "parent_refs"))
                if actual != expected:
                    raise ValueError("An external operation identity cannot be reused for different inputs")
                return request, attempt, self.chat["version_id"]
            db.execute(
                S.requests.insert().values(
                    id=request,
                    owner_id=self.owner,
                    conversation_id=self.conversation,
                    version_id=self.chat["version_id"],
                    source_sha256=checksum,
                    question=question,
                    context=context,
                    parent_refs=[r.model_dump(mode="json", exclude={"path"}) for r in refs],
                )
            )
            db.execute(S.attempts.insert().values(id=attempt, owner_id=self.owner, request_id=request, conversation_id=self.conversation, status="running"))
        return request, attempt, self.chat["version_id"]

    def external_task(self, attempt, task_id):
        with self.storage.engine.begin() as db:
            row = (
                db.execute(
                    select(S.attempts)
                    .where(S.attempts.c.id == attempt, S.attempts.c.owner_id == self.owner, S.attempts.c.conversation_id == self.conversation)
                    .with_for_update()
                )
                .mappings()
                .one()
            )
            if row["external_task_id"] and row["external_task_id"] != task_id:
                raise ValueError("This attempt already names a different external task")
            db.execute(S.attempts.update().where(S.attempts.c.id == attempt).values(external_task_id=task_id))

    def finish(self, attempt, status, detail):
        with self.storage.engine.begin() as db:
            db.execute(
                update(S.attempts)
                .where(
                    S.attempts.c.id == attempt,
                    S.attempts.c.owner_id == self.owner,
                    S.attempts.c.conversation_id == self.conversation,
                    S.attempts.c.status != "completed",
                )
                .values(status=status, detail=detail, ended_at=datetime.now(UTC))
            )

    def publish(self, a: Artifact, directory: Path):
        if (a.owner_id, a.conversation_id, a.dataset, a.dataset_version_id, a.source_sha256) != (
            self.owner,
            self.conversation,
            self.conversation,
            self.chat["version_id"],
            self.info["sha256"],
        ):
            raise ValueError("Artifact ownership or version mismatch")
        if not a.summary or not a.method or not a.findings or not a.limitations or any(not math.isfinite(v) for v in a.facts.values()):
            raise ValueError("Publication requires an evidence-backed explanation with finite numbers")
        if not (a.facts or a.figure or a.table or (a.origin == "analysis" and "result.json" in a.files)) or "facts.json" not in a.files:
            raise ValueError("Publication requires a numerical, tabular or visual product")
        if json.loads(safe_file(directory / "facts.json", directory, 5 * 1024 * 1024).read_text()) != a.facts:
            raise ValueError("Published numbers differ from the numerical artifact")
        if any(not grounded(value, list(a.facts.values())) for value in numbers_in(a.summary + " " + " ".join(a.findings))):
            raise ValueError("Published explanation contains unsupported numbers")
        if a.figure:
            from PIL import Image

            with Image.open(safe_file(directory / "figure.png", directory, 5 * 1024 * 1024)) as image:
                image.verify()
        if a.table and a.table not in a.files:
            raise ValueError("Result table is not registered")
        # Elements come from registered bytes, never model-invented selectors.
        a.elements = [ArtifactElement(id=key, label=key.replace("_", " "), kind="fact", file="facts.json", selector=key) for key in a.facts]
        if a.table:
            rows = table_rows(a.table, safe_file(directory / a.table, directory, 5 * 1024 * 1024).read_bytes())
            columns = rows[0] if rows else []
            a.elements.extend(
                ArtifactElement(id=f"column:{i}", label=name, kind="table_column", file=a.table, selector=str(i)) for i, name in enumerate(columns[:200])
            )
        if "figures.json" in a.files:
            from causal_agent.viz.spec import FigureSpec

            figures = [FigureSpec.model_validate(f) for f in json.loads(safe_file(directory / "figures.json", directory, 5 * 1024 * 1024).read_text())]
            a.elements.extend(
                ArtifactElement(id=f"series:{i}:{j}", label=f"{f.title}: {s.name}", kind="chart_series", file="figures.json", selector=f"{i}:{j}")
                for i, f in enumerate(figures)
                for j, s in enumerate(f.series)
            )
        if "artifact.json" in a.files:
            trace.write(directory / "artifact.json", a.model_dump_json(indent=2))
        with trace.span(
            "artifact:publish", inputs={"conversation_id": self.conversation, "artifact_id": a.id, "attempt_id": a.attempt_id, "origin": a.origin}
        ) as span:
            with self.storage.locked(self.owner, "publish:" + a.attempt_id) as db:
                attempt = (
                    db.execute(
                        select(S.attempts).where(
                            S.attempts.c.id == a.attempt_id,
                            S.attempts.c.owner_id == self.owner,
                            S.attempts.c.request_id == a.request_id,
                            S.attempts.c.conversation_id == self.conversation,
                        )
                    )
                    .mappings()
                    .first()
                )
                if attempt is not None and attempt["status"] == "completed":
                    saved = db.scalar(select(S.results.c.record).where(S.results.c.attempt_id == a.attempt_id, S.results.c.owner_id == self.owner))
                    if saved == a.model_dump(mode="json"):
                        return
                    raise ValueError("Published artifacts are immutable")
                if attempt is None or attempt["status"] not in {"running", "publishing"}:
                    raise ValueError("Attempt is not publishable")
                if db.scalar(select(S.artifacts.c.id).where(S.artifacts.c.id == a.id)) is not None:
                    raise ValueError("Published artifacts are immutable")
                request = db.execute(select(S.requests).where(S.requests.c.id == a.request_id)).mappings().one()
                if a.parent_refs != [ref["id"] for ref in request["parent_refs"]]:
                    raise ValueError("Artifact parent references differ from the supplied inputs")
                registered = []
                for name in a.files:
                    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", name):
                        raise ValueError("Invalid artifact filename")
                    path = safe_file(directory / name, directory, 5 * 1024 * 1024)
                    checksum, size = digest(path), path.stat().st_size
                    key = f"users/{self.owner}/conversations/{self.conversation}/artifacts/{a.id}/{name}"
                    registered.append(
                        dict(
                            artifact_id=a.id,
                            owner_id=self.owner,
                            conversation_id=self.conversation,
                            name=name,
                            object_key=key,
                            sha256=checksum,
                            size_bytes=size,
                        )
                    )
                db.execute(update(S.attempts).where(S.attempts.c.id == a.attempt_id).values(status="publishing", output_manifest=registered))
                db.commit()
                for item in registered:
                    with (directory / item["name"]).open("rb") as stream:
                        self.storage.objects.put(item["object_key"], stream, item["size_bytes"], item["sha256"])
                result_id = uid()
                db.execute(
                    S.results.insert().values(
                        id=result_id, owner_id=self.owner, conversation_id=self.conversation, attempt_id=a.attempt_id, record=a.model_dump(mode="json")
                    )
                )
                db.execute(S.artifacts.insert().values(id=a.id, owner_id=self.owner, conversation_id=self.conversation, result_id=result_id))
                db.execute(S.files.insert(), registered)
                db.execute(update(S.attempts).where(S.attempts.c.id == a.attempt_id).values(status="completed", ended_at=datetime.now(UTC)))
                db.commit()
                span.end(outputs={"artifact_id": a.id, "result_id": result_id, "files": [r["name"] for r in registered]})

    def permitted(self):
        same_dataset = S.artifacts.c.conversation_id.in_(
            select(S.conversations.c.id).where(S.conversations.c.owner_id == self.owner, S.conversations.c.dataset_id == self.chat["dataset_id"])
        )
        return same_dataset & or_(
            S.artifacts.c.conversation_id == self.conversation,
            S.artifacts.c.id.in_(
                select(S.selections.c.artifact_id).where(S.selections.c.conversation_id == self.conversation, S.selections.c.owner_id == self.owner)
            ),
        )

    def available(self, artifact_id):
        with self.storage.engine.connect() as db:
            record = db.scalar(
                select(S.results.c.record)
                .join(S.artifacts, S.artifacts.c.result_id == S.results.c.id)
                .join(S.conversations, S.conversations.c.id == S.artifacts.c.conversation_id)
                .where(
                    S.artifacts.c.id == artifact_id,
                    S.artifacts.c.owner_id == self.owner,
                    S.conversations.c.owner_id == self.owner,
                    S.conversations.c.dataset_id == self.chat["dataset_id"],
                )
            )
        if record is None:
            raise Missing("Artifact not found in this owned dataset.")
        return Artifact.model_validate(record)

    def reuse(self, db, artifact_id, operation_id):
        self.available(artifact_id)
        db.execute(
            insert(S.selections)
            .values(artifact_id=artifact_id, owner_id=self.owner, conversation_id=self.conversation, operation_id=operation_id)
            .on_conflict_do_nothing(index_elements=["conversation_id", "artifact_id"])
        )

    def earlier(self, limit=30, offset=0):
        with self.storage.engine.connect() as db:
            records = db.scalars(
                select(S.results.c.record)
                .join(S.artifacts, S.artifacts.c.result_id == S.results.c.id)
                .join(S.conversations, S.conversations.c.id == S.artifacts.c.conversation_id)
                .where(
                    S.artifacts.c.owner_id == self.owner,
                    S.conversations.c.owner_id == self.owner,
                    S.conversations.c.dataset_id == self.chat["dataset_id"],
                    ~self.permitted(),
                )
                .order_by(S.artifacts.c.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return [Artifact.model_validate(r) for r in records]

    def list(self, dataset, limit=50, offset=0):
        if dataset != self.conversation:
            return []
        with self.storage.engine.connect() as db:
            records = db.scalars(
                select(S.results.c.record)
                .join(S.artifacts, S.artifacts.c.result_id == S.results.c.id)
                .where(S.artifacts.c.owner_id == self.owner, self.permitted())
                .order_by(S.artifacts.c.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return [Artifact.model_validate(record) for record in records]

    def load(self, dataset, artifact_id):
        if dataset != self.conversation:
            raise Missing("Artifact not found.")
        with self.storage.engine.connect() as db:
            record = db.scalar(
                select(S.results.c.record)
                .join(S.artifacts, S.artifacts.c.result_id == S.results.c.id)
                .where(S.artifacts.c.owner_id == self.owner, self.permitted(), S.artifacts.c.id == artifact_id)
            )
        if record is None:
            raise Missing("Artifact not found.")
        return Artifact.model_validate(record)

    def read_file(self, artifact_id, name):
        self.load(self.conversation, artifact_id)
        with self.storage.engine.connect() as db:
            row = (
                db.execute(
                    select(S.files).where(
                        S.files.c.artifact_id == artifact_id,
                        S.files.c.name == name,
                        S.files.c.owner_id == self.owner,
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            raise Missing("Artifact file not found.")
        body = self.storage.objects.open(row["object_key"])
        try:
            data = body.read(5 * 1024 * 1024 + 1)
        finally:
            body.close()
        if len(data) != row["size_bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise OSError("Published artifact checksum mismatch")
        return data

    def materialize(self, a, directory):
        directory.mkdir(parents=True, exist_ok=True)
        for name in a.files:
            trace.write(directory / name, self.read_file(a.id, name))


def cleanup(storage):
    """Reconcile expired unpublished payloads under the same lock as publication."""
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    with storage.engine.connect() as db:
        rows = list(db.execute(select(S.attempts).where(S.attempts.c.status != "completed", S.attempts.c.created_at < cutoff)).mappings())
    for row in rows:
        with storage.locked(row["owner_id"], "publish:" + row["id"]) as db:
            attempt = db.execute(select(S.attempts).where(S.attempts.c.id == row["id"])).mappings().one()
            if attempt["status"] == "completed":
                continue
            prefix = f"users/{row['owner_id']}/conversations/{row['conversation_id']}/artifacts/"
            for item in attempt["output_manifest"]:
                if not item["object_key"].startswith(prefix):
                    raise ValueError("Invalid unpublished object ownership")
                storage.objects.delete(item["object_key"])
            db.execute(update(S.attempts).where(S.attempts.c.id == row["id"]).values(status="failed", detail="Expired unfinished attempt", output_manifest=[]))
            db.commit()
    return len(rows)
