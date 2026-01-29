"""Authoritative relational model. Changes require an Alembic migration and docs/data-model.md update."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(254), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("owner_id", "submission_id", name="uq_dataset_submission"),)
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    submission_id: Mapped[UUID]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint("dataset_id", "number", name="uq_dataset_version_number"),
        CheckConstraint("number > 0 AND size_bytes > 0 AND row_count > 0", name="ck_version_positive"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(ForeignKey("datasets.id"), index=True)
    number: Mapped[int]
    filename: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    row_count: Mapped[int]
    columns: Mapped[list[str]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class SourceMaterial(Base):
    __tablename__ = "source_materials"
    __table_args__ = (
        UniqueConstraint("version_id", "submission_id", name="uq_source_submission"),
        UniqueConstraint("id", "version_id", name="uq_source_version"),
        CheckConstraint("size_bytes > 0 AND character_count > 0", name="ck_source_positive"),
        CheckConstraint("kind IN ('file', 'note')", name="ck_source_kind"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    version_id: Mapped[UUID] = mapped_column(ForeignKey("dataset_versions.id"), index=True)
    submission_id: Mapped[UUID]
    title: Mapped[str] = mapped_column(String(200))
    filename: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(20))
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    character_count: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextPack(Base):
    __tablename__ = "context_packs"
    __table_args__ = (
        UniqueConstraint("id", "version_id", name="uq_pack_version"),
        ForeignKeyConstraint(
            ["current_revision_id", "id"],
            ["context_revisions.id", "context_revisions.pack_id"],
            name="fk_pack_current",
            use_alter=True,
        ),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    version_id: Mapped[UUID] = mapped_column(ForeignKey("dataset_versions.id"), unique=True)
    current_revision_id: Mapped[UUID | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ContextRevision(Base):
    __tablename__ = "context_revisions"
    __table_args__ = (
        UniqueConstraint("pack_id", "number", name="uq_context_revision_number"),
        UniqueConstraint("pack_id", "submission_id", name="uq_context_revision_submission"),
        UniqueConstraint("id", "version_id", name="uq_revision_version"),
        UniqueConstraint("id", "pack_id", name="uq_revision_pack"),
        ForeignKeyConstraint(
            ["pack_id", "version_id"],
            ["context_packs.id", "context_packs.version_id"],
            name="fk_revision_pack_version",
        ),
        ForeignKeyConstraint(
            ["parent_id", "pack_id"],
            ["context_revisions.id", "context_revisions.pack_id"],
            name="fk_revision_parent_pack",
        ),
        CheckConstraint("number > 0 AND state IN ('draft', 'reviewed')", name="ck_context_revision"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    pack_id: Mapped[UUID] = mapped_column(ForeignKey("context_packs.id"), index=True)
    version_id: Mapped[UUID] = mapped_column(ForeignKey("dataset_versions.id"))
    parent_id: Mapped[UUID | None] = mapped_column(ForeignKey("context_revisions.id"))
    number: Mapped[int]
    submission_id: Mapped[UUID]
    request_hash: Mapped[str] = mapped_column(String(64))
    schema_version: Mapped[str] = mapped_column(String(20), default="2")
    state: Mapped[str] = mapped_column(String(20))
    author_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    content: Mapped[dict[str, object]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class RevisionSource(Base):
    __tablename__ = "context_revision_sources"
    __table_args__ = (
        ForeignKeyConstraint(
            ["revision_id", "version_id"],
            ["context_revisions.id", "context_revisions.version_id"],
            name="fk_revision_source_version",
        ),
        ForeignKeyConstraint(
            ["source_id", "version_id"],
            ["source_materials.id", "source_materials.version_id"],
            name="fk_source_revision_version",
        ),
    )
    revision_id: Mapped[UUID] = mapped_column(ForeignKey("context_revisions.id"), primary_key=True)
    source_id: Mapped[UUID] = mapped_column(ForeignKey("source_materials.id"), primary_key=True)
    version_id: Mapped[UUID] = mapped_column(ForeignKey("dataset_versions.id"))


class ContextPreparation(Base):
    __tablename__ = "context_preparations"
    __table_args__ = (
        UniqueConstraint("pack_id", "submission_id", name="uq_preparation_submission"),
        ForeignKeyConstraint(
            ["base_revision_id", "pack_id"],
            ["context_revisions.id", "context_revisions.pack_id"],
            name="fk_preparation_base_pack",
        ),
        ForeignKeyConstraint(
            ["result_revision_id", "pack_id"],
            ["context_revisions.id", "context_revisions.pack_id"],
            name="fk_preparation_result_pack",
        ),
        CheckConstraint("calls >= 0 AND calls <= 4", name="ck_preparation_calls"),
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')", name="ck_preparation_status"
        ),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    pack_id: Mapped[UUID] = mapped_column(ForeignKey("context_packs.id"), index=True)
    base_revision_id: Mapped[UUID | None] = mapped_column(ForeignKey("context_revisions.id"))
    submission_id: Mapped[UUID]
    request_hash: Mapped[str] = mapped_column(String(64))
    author_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    inputs: Mapped[dict[str, object]] = mapped_column(JSONB)
    steps: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    stage: Mapped[str] = mapped_column(String(40), default="extract")
    calls: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(String(4000))
    job_id: Mapped[int | None] = mapped_column(BigInteger)
    result_revision_id: Mapped[UUID | None] = mapped_column(ForeignKey("context_revisions.id"))
    trace_id: Mapped[UUID]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        UniqueConstraint("version_id", "submission_id", name="uq_conversation_submission"),
        ForeignKeyConstraint(
            ["context_revision_id", "version_id"],
            ["context_revisions.id", "context_revisions.version_id"],
            name="fk_conversation_context_version",
        ),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    version_id: Mapped[UUID] = mapped_column(ForeignKey("dataset_versions.id"), index=True)
    context_revision_id: Mapped[UUID | None]
    submission_id: Mapped[UUID]
    title: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ChatTurn(Base):
    __tablename__ = "chat_turns"
    __table_args__ = (
        UniqueConstraint("conversation_id", "number", name="uq_chat_turn_number"),
        UniqueConstraint("conversation_id", "submission_id", name="uq_chat_turn_submission"),
        CheckConstraint("number > 0 AND calls BETWEEN 0 AND 24", name="ck_chat_turn_budget"),
        CheckConstraint("status IN ('queued','running','succeeded','failed')", name="ck_chat_turn_status"),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("conversations.id"), index=True)
    number: Mapped[int]
    submission_id: Mapped[UUID]
    request_hash: Mapped[str] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(String(8000))
    inputs: Mapped[dict[str, object]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    stage: Mapped[str] = mapped_column(String(40), default="interpret")
    calls: Mapped[int] = mapped_column(default=0)
    receipts: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict)
    reply: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(String(4000))
    job_id: Mapped[int | None] = mapped_column(BigInteger)
    trace_id: Mapped[UUID]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ChatArtifact(Base):
    __tablename__ = "chat_artifacts"
    id: Mapped[UUID] = mapped_column(primary_key=True)
    turn_id: Mapped[UUID] = mapped_column(ForeignKey("chat_turns.id"), unique=True)
    record: Mapped[dict[str, object]] = mapped_column(JSONB)
    files: Mapped[dict[str, object]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ChatContextSelection(Base):
    __tablename__ = "chat_context_selections"
    turn_id: Mapped[UUID] = mapped_column(ForeignKey("chat_turns.id"), primary_key=True)
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("conversations.id"), index=True)
    revision_id: Mapped[UUID] = mapped_column(ForeignKey("context_revisions.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (
        UniqueConstraint("conversation_id", "submission_id", name="uq_analysis_submission"),
        ForeignKeyConstraint(
            ["context_revision_id", "version_id"],
            ["context_revisions.id", "context_revisions.version_id"],
            name="fk_analysis_context_version",
        ),
        CheckConstraint("calls >= 0 AND shared_calls >= 0 AND (model_limit IS NULL OR calls <= model_limit)", name="ck_analysis_budget"),
        CheckConstraint(
            "status IN ('queued','running','awaiting_approval','succeeded','blocked','failed')",
            name="ck_analysis_status",
        ),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("conversations.id"), index=True)
    version_id: Mapped[UUID] = mapped_column(ForeignKey("dataset_versions.id"))
    context_revision_id: Mapped[UUID | None]
    submission_id: Mapped[UUID]
    author_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    question: Mapped[str] = mapped_column(String(8000))
    inputs: Mapped[dict[str, object]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(24), default="queued")
    stage: Mapped[str] = mapped_column(String(40), default="bind")
    calls: Mapped[int] = mapped_column(default=0)
    shared_calls: Mapped[int] = mapped_column(default=0, server_default="0")
    model_limit: Mapped[int | None] = mapped_column()
    receipts: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict)
    snapshot: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    snapshot_hash: Mapped[str | None] = mapped_column(String(64))
    approved_hash: Mapped[str | None] = mapped_column(String(64))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(String(4000))
    job_id: Mapped[int | None] = mapped_column(BigInteger)
    trace_id: Mapped[UUID]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AnalysisEvent(Base):
    __tablename__ = "analysis_events"
    __table_args__ = (UniqueConstraint("run_id", "key", name="uq_analysis_event_key"),)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("analysis_runs.id"), primary_key=True)
    sequence: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(100))
    stage: Mapped[str] = mapped_column(String(40))
    kind: Mapped[str] = mapped_column(String(30))
    summary: Mapped[str] = mapped_column(String(4000))
    detail: Mapped[dict[str, object]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AnalysisTask(Base):
    __tablename__ = "analysis_tasks"
    __table_args__ = (
        UniqueConstraint("run_id", "key", name="uq_analysis_task_key"),
        CheckConstraint("max_attempts BETWEEN 1 AND 2 AND generation >= 0", name="ck_analysis_task_allowance"),
        CheckConstraint("budget_kind IN ('column','relationship','shared')", name="ck_analysis_task_budget_kind"),
        CheckConstraint(
            "status IN ('pending','running','accepted','rejected','failed','blocked','exhausted','expanded')",
            name="ck_analysis_task_status",
        ),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("analysis_runs.id"), index=True)
    key: Mapped[str] = mapped_column(String(200))
    checkpoint: Mapped[str] = mapped_column(String(40))
    kind: Mapped[str] = mapped_column(String(40))
    budget_kind: Mapped[str] = mapped_column(String(20))
    column_ids: Mapped[list[str]] = mapped_column(JSONB)
    obligation_keys: Mapped[list[str]] = mapped_column(JSONB)
    scope: Mapped[dict[str, object]] = mapped_column(JSONB)
    reason: Mapped[str] = mapped_column(String(4000))
    dependency_receipt_ids: Mapped[list[str]] = mapped_column(JSONB)
    input_hash: Mapped[str] = mapped_column(String(64))
    packet: Mapped[dict[str, object]] = mapped_column(JSONB)
    max_attempts: Mapped[int] = mapped_column(default=2)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    generation: Mapped[int] = mapped_column(default=0)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    parent_id: Mapped[UUID | None] = mapped_column(ForeignKey("analysis_tasks.id"))
    agent: Mapped[str | None] = mapped_column(String(40))
    model_limit: Mapped[int | None] = mapped_column()
    submissions: Mapped[int] = mapped_column(default=0, server_default="0")
    draft: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, server_default="{}")
    handoff: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    handoff_hash: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AnalysisTaskAttempt(Base):
    __tablename__ = "analysis_task_attempts"
    __table_args__ = (
        UniqueConstraint("run_id", "key", name="uq_analysis_attempt_key"),
        UniqueConstraint("task_id", "number", name="uq_analysis_task_attempt"),
        CheckConstraint("number > 0 AND generation >= 0", name="ck_analysis_attempt_number"),
        CheckConstraint(
            "status IN ('reserved','accepted','rejected','failed','uncertain','blocked','exhausted','expanded')",
            name="ck_analysis_attempt_status",
        ),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("analysis_runs.id"), index=True)
    task_id: Mapped[UUID | None] = mapped_column(ForeignKey("analysis_tasks.id"), index=True)
    key: Mapped[str] = mapped_column(String(240))
    number: Mapped[int]
    generation: Mapped[int]
    status: Mapped[str] = mapped_column(String(20), default="reserved")
    input_hash: Mapped[str] = mapped_column(String(64))
    request: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    request_hash: Mapped[str | None] = mapped_column(String(64))
    receipt: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    raw_output: Mapped[str | None] = mapped_column(String)
    provider_content: Mapped[dict[str, object] | None] = mapped_column(JSONB)
    tool_receipts: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, server_default="{}")
    metrics: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
