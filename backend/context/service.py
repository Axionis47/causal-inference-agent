"""Owned transactions for sources, revisions and preparation submission."""

import hashlib
import json
from typing import Any, Literal
from uuid import UUID, uuid4, uuid5

from fastapi import HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from backend.config import settings
from backend.context.contracts import (
    ContextContent,
    PreparationView,
    PrepareInput,
    RetrievedContext,
    RevisionSummary,
    RevisionView,
    SaveInput,
    SourceText,
    SourceView,
    WorkspaceView,
    empty_content,
    statement_entries,
)
from backend.context.domain import apply_brief, apply_edits, project, validate_evidence
from backend.models import (
    ContextPack,
    ContextPreparation,
    ContextRevision,
    Dataset,
    DatasetVersion,
    RevisionSource,
    SourceMaterial,
    now,
)
from backend.storage import LocalFiles


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def owned_version(db: Session, owner: UUID, version_id: UUID) -> DatasetVersion:
    version = db.scalar(
        select(DatasetVersion).join(Dataset).where(DatasetVersion.id == version_id, Dataset.owner_id == owner)
    )
    if version is None:
        raise HTTPException(404, "Dataset version not found.")
    return version


def locked_pack(db: Session, version_id: UUID) -> ContextPack:
    db.execute(
        text("SELECT pg_advisory_xact_lock(:key)"),
        {"key": int.from_bytes(version_id.bytes[:8], "big", signed=True)},
    )
    pack = db.scalar(
        select(ContextPack)
        .where(ContextPack.version_id == version_id)
        .execution_options(populate_existing=True)
    )
    if pack is None:
        pack = ContextPack(id=uuid5(version_id, "context"), version_id=version_id)
        db.add(pack)
        db.flush()
    return pack


def source_record(db: Session, version: UUID, source_id: UUID) -> SourceMaterial:
    source = db.scalar(
        select(SourceMaterial).where(SourceMaterial.id == source_id, SourceMaterial.version_id == version)
    )
    if source is None:
        raise HTTPException(404, "Source not found.")
    return source


def source_text(db: Session, storage: LocalFiles, version: UUID, source_id: UUID) -> SourceText:
    source = source_record(db, version, source_id)
    return SourceText(
        source=SourceView.model_validate(source),
        text=storage.read(source.storage_key, source.sha256).decode("utf-8"),
    )


def upload_source(
    db: Session,
    storage: LocalFiles,
    version: UUID,
    submission: UUID,
    title: str,
    filename: str,
    kind: Literal["file", "note"],
    content: bytes,
) -> SourceView:
    title = title.strip()
    filename = filename.replace("\\", "/").split("/")[-1]
    if (
        not title
        or len(title) > 200
        or not filename
        or len(filename) > 255
        or any(ord(c) < 32 for c in filename)
    ):
        raise ValueError("Supply a title and valid filename.")
    if not filename.lower().endswith((".txt", ".md")):
        raise ValueError("Choose a .txt or .md source.")
    if not content or len(content) > settings().context_source_bytes:
        raise ValueError("Choose a nonempty source up to 1 MiB.")
    try:
        decoded = content.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("Sources must be UTF-8 text.") from exc
    if "\x00" in decoded or not decoded.strip("\ufeff \t\r\n"):
        raise ValueError("The source must contain readable text without NUL bytes.")
    checksum = hashlib.sha256(content).hexdigest()
    locked_pack(db, version)
    identity = uuid5(version, f"source:{submission}")
    existing = db.get(SourceMaterial, identity)
    if existing:
        if (existing.title, existing.filename, existing.kind, existing.sha256) != (
            title,
            filename,
            kind,
            checksum,
        ):
            raise HTTPException(409, "This submission was saved with different source content.")
        db.commit()
        return SourceView.model_validate(existing)
    count = (
        db.scalar(
            select(func.count()).select_from(SourceMaterial).where(SourceMaterial.version_id == version)
        )
        or 0
    )
    if count >= settings().context_max_sources:
        raise ValueError("This dataset version already has ten sources.")
    key = f"context/{version}/{identity}/{checksum}"
    storage.put(key, content)
    source = SourceMaterial(
        id=identity,
        version_id=version,
        submission_id=submission,
        title=title,
        filename=filename,
        kind=kind,
        storage_key=key,
        sha256=checksum,
        size_bytes=len(content),
        character_count=len(decoded),
    )
    db.add(source)
    db.commit()
    return SourceView.model_validate(source)


def revision_record(db: Session, version: UUID, revision_id: UUID) -> ContextRevision:
    record = db.scalar(
        select(ContextRevision).where(
            ContextRevision.id == revision_id, ContextRevision.version_id == version
        )
    )
    if record is None:
        raise HTTPException(404, "Context revision not found.")
    return record


def revision_sources(db: Session, record: ContextRevision) -> list[SourceMaterial]:
    return list(
        db.scalars(
            select(SourceMaterial)
            .join(RevisionSource, RevisionSource.source_id == SourceMaterial.id)
            .where(RevisionSource.revision_id == record.id)
            .order_by(SourceMaterial.created_at, SourceMaterial.id)
        )
    )


def revision_view(db: Session, record: ContextRevision) -> RevisionView:
    return RevisionView(
        **RevisionSummary.model_validate(record).model_dump(),
        version_id=record.version_id,
        schema_version=record.schema_version,
        content=ContextContent.model_validate(record.content),
        sources={str(s.id): SourceView.model_validate(s) for s in revision_sources(db, record)},
    )


def workspace(db: Session, version: DatasetVersion) -> WorkspaceView:
    pack = db.scalar(select(ContextPack).where(ContextPack.version_id == version.id))
    return WorkspaceView(
        version_id=version.id,
        current_revision_id=pack.current_revision_id if pack else None,
        sources=[
            SourceView.model_validate(s)
            for s in db.scalars(
                select(SourceMaterial)
                .where(SourceMaterial.version_id == version.id)
                .order_by(SourceMaterial.created_at, SourceMaterial.id)
            )
        ],
        revisions=[
            RevisionSummary.model_validate(r)
            for r in db.scalars(
                select(ContextRevision)
                .where(ContextRevision.version_id == version.id)
                .order_by(ContextRevision.number.desc())
            )
        ],
        preparations=[
            PreparationView.model_validate(p)
            for p in db.scalars(
                select(ContextPreparation)
                .where(ContextPreparation.pack_id == pack.id)
                .order_by(ContextPreparation.created_at.desc())
                .limit(20)
            )
        ]
        if pack
        else [],
        columns=[c.identity for c in empty_content(version.id, version.columns).columns.values()],
    )


def check_parent(pack: ContextPack, expected: UUID | None) -> None:
    if pack.current_revision_id != expected:
        raise HTTPException(
            409,
            "Context changed. Reload the current revision before saving; your edits have not been merged.",
        )


def append_revision(
    db: Session,
    pack: ContextPack,
    parent: ContextRevision | None,
    identity: UUID,
    submission: UUID,
    request_hash: str,
    author: UUID,
    content: ContextContent,
    source_ids: list[UUID],
    state: Literal["draft", "reviewed"],
) -> ContextRevision:
    record = ContextRevision(
        id=identity,
        pack_id=pack.id,
        version_id=pack.version_id,
        parent_id=parent.id if parent else None,
        number=parent.number + 1 if parent else 1,
        submission_id=submission,
        request_hash=request_hash,
        author_id=author,
        state=state,
        content=content.model_dump(mode="json"),
    )
    db.add(record)
    db.flush()
    db.add_all(
        [
            RevisionSource(revision_id=record.id, source_id=sid, version_id=pack.version_id)
            for sid in source_ids
        ]
    )
    db.flush()
    pack.current_revision_id = record.id
    return record


def save_revision(
    db: Session,
    storage: LocalFiles,
    version: DatasetVersion,
    owner: UUID,
    body: SaveInput,
    *,
    commit: bool = True,
    brief_source_text: str = "Applied analysis brief.",
) -> RevisionView:
    pack = locked_pack(db, version.id)
    digest = fingerprint(body.model_dump(mode="json"))
    existing = db.scalar(
        select(ContextRevision).where(
            ContextRevision.pack_id == pack.id, ContextRevision.submission_id == body.submission_id
        )
    )
    if existing:
        if existing.request_hash != digest:
            raise HTTPException(409, "This submission already saved different changes.")
        if commit:
            db.commit()
        return revision_view(db, existing)
    check_parent(pack, body.expected_revision_id)
    parent = revision_record(db, version.id, body.expected_revision_id)
    identity = uuid5(pack.id, f"revision:{body.submission_id}")
    content = apply_edits(ContextContent.model_validate(parent.content), body.edits, owner, identity)
    content = apply_brief(content, body.analysis_brief, body.edits, owner, identity, source_text=brief_source_text)
    if body.leave_remaining_unresolved:
        from backend.context.contracts import SourceReference

        for _, statement in statement_entries(content):
            if statement.status == "proposed":
                statement.status = "unresolved"
                statement.review_source = SourceReference(
                    kind="review",
                    id=identity,
                    author_id=owner,
                    text="Explicitly left unresolved during review.",
                )
    if body.state == "reviewed" and any(s.status == "proposed" for _, s in statement_entries(content)):
        raise ValueError("Review the remaining proposals or explicitly leave them unresolved.")
    if body.state == parent.state and content == ContextContent.model_validate(parent.content):
        raise ValueError("There are no context or review-state changes to save.")
    sources = revision_sources(db, parent)
    validate_evidence(
        content, {str(s.id): storage.read(s.storage_key, s.sha256).decode("utf-8") for s in sources}
    )
    record = append_revision(
        db,
        pack,
        parent,
        identity,
        body.submission_id,
        digest,
        owner,
        content,
        [s.id for s in sources],
        body.state,
    )
    if commit:
        db.commit()
    return revision_view(db, record)


def prepare(
    db: Session, storage: LocalFiles, version: DatasetVersion, owner: UUID, body: PrepareInput
) -> PreparationView:
    from backend.context.extraction import ASSESS, EXTRACT, schema_digest
    from backend.context.jobs import enqueue

    pack = locked_pack(db, version.id)
    digest = fingerprint(body.model_dump(mode="json"))
    identity = uuid5(pack.id, f"prepare:{body.submission_id}")
    existing = db.get(ContextPreparation, identity)
    if existing:
        if existing.request_hash != digest:
            raise HTTPException(409, "This preparation submission has different inputs.")
        db.commit()
        return PreparationView.model_validate(existing)
    check_parent(pack, body.expected_revision_id)
    if len(body.source_ids) != len(set(body.source_ids)):
        raise ValueError("Select each source once.")
    parent = revision_record(db, version.id, body.expected_revision_id) if body.expected_revision_id else None
    previous_sources = revision_sources(db, parent) if parent else []
    old_ids = {s.id for s in previous_sources}
    if old_ids.intersection(body.source_ids):
        raise ValueError("Select new sources; included sources are already preserved in the pack.")
    sources = [source_record(db, version.id, sid) for sid in body.source_ids]
    all_sources = [*previous_sources, *sources]
    if (
        len(all_sources) > settings().context_max_sources
        or sum(s.character_count for s in all_sources) > settings().context_max_characters
    ):
        raise ValueError(
            "Preparation supports at most ten sources and 100,000 decoded characters. Sources were not truncated."
        )
    if db.scalar(
        select(ContextPreparation.id).where(
            ContextPreparation.pack_id == pack.id, ContextPreparation.status.in_(["queued", "running"])
        )
    ):
        raise HTTPException(409, "A preparation is already queued or running for this version.")
    operation = ContextPreparation(
        id=identity,
        pack_id=pack.id,
        base_revision_id=body.expected_revision_id,
        submission_id=body.submission_id,
        request_hash=digest,
        author_id=owner,
        trace_id=uuid4(),
        inputs={
            "version_id": str(version.id),
            "columns": version.columns,
            "sources": {
                str(s.id): {
                    "title": s.title,
                    "sha256": s.sha256,
                    "text": storage.read(s.storage_key, s.sha256).decode("utf-8"),
                }
                for s in sources
            },
            "source_ids": [str(s.id) for s in all_sources],
            "model": settings().gemini_model,
            "extractor_version": "keyed-context-10",
            "extract_prompt": EXTRACT,
            "assess_prompt": ASSESS,
            "schema_digest": schema_digest(),
        },
    )
    db.add(operation)
    db.flush()
    operation.job_id = enqueue(db, operation.id)
    db.commit()
    return PreparationView.model_validate(operation)


def preparation_record(db: Session, version: UUID, identity: UUID) -> ContextPreparation:
    operation = db.scalar(
        select(ContextPreparation)
        .join(ContextPack)
        .where(ContextPreparation.id == identity, ContextPack.version_id == version)
    )
    if operation is None:
        raise HTTPException(404, "Preparation not found.")
    return operation


def retry_preparation(db: Session, version: UUID, identity: UUID) -> PreparationView:
    from backend.context.jobs import enqueue

    pack = locked_pack(db, version)
    operation = preparation_record(db, version, identity)
    if operation.status != "failed":
        return PreparationView.model_validate(operation)
    check_parent(pack, operation.base_revision_id)
    if operation.calls >= 4 or operation.stage == "rejected":
        raise HTTPException(
            409,
            "This preparation exhausted its bounded attempts. Start a new preparation after reviewing the failure.",
        )
    if db.scalar(
        select(ContextPreparation.id).where(
            ContextPreparation.pack_id == pack.id, ContextPreparation.status.in_(["queued", "running"])
        )
    ):
        raise HTTPException(409, "Another preparation is active.")
    operation.status, operation.error, operation.updated_at = "queued", None, now()
    operation.job_id = enqueue(db, operation.id)
    db.commit()
    return PreparationView.model_validate(operation)


def read_context(db: Session, version: UUID, revision_id: UUID, keys: list[str]) -> RetrievedContext:
    from backend.context.domain import related_to

    record = revision_record(db, version, revision_id)
    source_map = {str(s.id): SourceView.model_validate(s) for s in revision_sources(db, record)}
    semantic_keys = [k for k in keys if k != "sources" and not k.startswith("sources.")]
    if semantic_keys or not keys:
        content, included, source_ids = project(ContextContent.model_validate(record.content), semantic_keys)
    else:
        content, included, source_ids = ContextContent(), [], set()
    for key in keys:
        if key == "sources":
            source_ids.update(source_map)
        elif key.startswith("sources."):
            sid = key.removeprefix("sources.")
            if sid not in source_map:
                raise ValueError("Unknown source key in this revision.")
            source_ids.add(sid)
    if not keys:
        source_ids.update(source_map)
    included.extend(f"sources.{sid}" for sid in sorted(source_ids))
    return RetrievedContext(
        revision_id=record.id,
        version_id=version,
        requested_keys=keys,
        included_keys=sorted(set(included)),
        content=content,
        related_to=related_to(content),
        sources={sid: source_map[sid] for sid in source_ids},
    )
