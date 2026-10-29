"""Conversation -> exploration request -> attempt -> result -> artifact/files."""

from sqlalchemy import BigInteger, Boolean, CheckConstraint, Column, DateTime, ForeignKeyConstraint, MetaData, String, Table, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB

from causal_agent.server.context.schema import reviews, revisions
from causal_agent.server.ingestion import schema as I

metadata = MetaData(schema="chat")
for table in I.metadata.tables.values():
    table.to_metadata(metadata, schema="ingestion")
for table in (revisions, reviews):
    table.to_metadata(metadata, schema="context")


def identity():
    return [Column("id", String(36), primary_key=True), Column("owner_id", String(36), nullable=False), UniqueConstraint("id", "owner_id")]


def parent(column, table):
    return ForeignKeyConstraint([column, "owner_id"], [f"chat.{table}.id", f"chat.{table}.owner_id"])


def stamp():
    return Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now())


conversations = Table(
    "conversations",
    metadata,
    *identity(),
    Column("dataset_id", String(36), nullable=False),
    Column("version_id", String(36), nullable=False),
    Column("title", Text, nullable=False),
    Column("context", Text, nullable=False),
    stamp(),
    ForeignKeyConstraint(["dataset_id", "owner_id"], ["ingestion.datasets.id", "ingestion.datasets.owner_id"]),
    ForeignKeyConstraint(["version_id", "owner_id", "dataset_id"], ["ingestion.versions.id", "ingestion.versions.owner_id", "ingestion.versions.dataset_id"]),
    UniqueConstraint("id", "owner_id", "version_id"),
)
requests = Table(
    "exploration_requests",
    metadata,
    *identity(),
    Column("conversation_id", String(36), nullable=False),
    Column("version_id", String(36), nullable=False),
    Column("source_sha256", String(64), nullable=False),
    Column("question", Text, nullable=False),
    Column("context", Text, nullable=False),
    Column("parent_refs", JSONB, nullable=False),
    stamp(),
    ForeignKeyConstraint(
        ["conversation_id", "owner_id", "version_id"], ["chat.conversations.id", "chat.conversations.owner_id", "chat.conversations.version_id"]
    ),
    UniqueConstraint("id", "owner_id", "conversation_id"),
)
attempts = Table(
    "attempts",
    metadata,
    *identity(),
    Column("request_id", String(36), nullable=False),
    Column("conversation_id", String(36), nullable=False),
    Column("status", String(20), nullable=False),
    Column("output_manifest", JSONB, nullable=False, server_default="[]"),
    Column("detail", Text, nullable=False, server_default=""),
    Column("external_task_id", Text),
    stamp(),
    Column("ended_at", DateTime(timezone=True)),
    ForeignKeyConstraint(
        ["request_id", "owner_id", "conversation_id"],
        ["chat.exploration_requests.id", "chat.exploration_requests.owner_id", "chat.exploration_requests.conversation_id"],
    ),
    UniqueConstraint("id", "owner_id", "conversation_id"),
    CheckConstraint("status IN ('running', 'publishing', 'completed', 'failed', 'needs_input')"),
)
results = Table(
    "exploration_results",
    metadata,
    *identity(),
    Column("attempt_id", String(36), nullable=False, unique=True),
    Column("conversation_id", String(36), nullable=False),
    Column("record", JSONB, nullable=False),
    stamp(),
    ForeignKeyConstraint(["attempt_id", "owner_id", "conversation_id"], ["chat.attempts.id", "chat.attempts.owner_id", "chat.attempts.conversation_id"]),
    UniqueConstraint("id", "owner_id", "conversation_id"),
)
artifacts = Table(
    "artifacts",
    metadata,
    *identity(),
    Column("conversation_id", String(36), nullable=False),
    Column("result_id", String(36), nullable=False),
    stamp(),
    ForeignKeyConstraint(
        ["result_id", "owner_id", "conversation_id"],
        ["chat.exploration_results.id", "chat.exploration_results.owner_id", "chat.exploration_results.conversation_id"],
    ),
    UniqueConstraint("id", "owner_id", "conversation_id"),
)
files = Table(
    "artifact_files",
    metadata,
    Column("artifact_id", String(36), primary_key=True),
    Column("name", String(120), primary_key=True),
    Column("owner_id", String(36), nullable=False),
    Column("conversation_id", String(36), nullable=False),
    Column("object_key", Text, nullable=False, unique=True),
    Column("sha256", String(64), nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    ForeignKeyConstraint(["artifact_id", "owner_id", "conversation_id"], ["chat.artifacts.id", "chat.artifacts.owner_id", "chat.artifacts.conversation_id"]),
    CheckConstraint("size_bytes >= 0"),
)

state = Table(
    "state",
    metadata,
    Column("conversation_id", String(36), primary_key=True),
    Column("owner_id", String(36), nullable=False),
    Column("version_id", String(36), nullable=False),
    Column("review_id", String(36), nullable=False, unique=True),
    Column("revision_id", String(36), nullable=False),
    Column("question", Text, nullable=False),
    Column("focus", JSONB),
    Column("fast_track", Boolean, nullable=False, server_default="false"),
    Column("status", String(20), nullable=False, server_default="idle"),
    Column("active_operation", String(36)),
    Column("prompt", Text, nullable=False, server_default=""),
    Column("pending", JSONB, nullable=False, server_default="[]"),
    Column("preparation", JSONB),
    Column("frame", JSONB),
    Column("readiness", JSONB, nullable=False, server_default='{"status":"unprepared","missing":0}'),
    Column("error", Text),
    parent("conversation_id", "conversations"),
    ForeignKeyConstraint(
        ["conversation_id", "owner_id", "version_id"], ["chat.conversations.id", "chat.conversations.owner_id", "chat.conversations.version_id"]
    ),
    ForeignKeyConstraint(["review_id", "owner_id", "version_id"], ["context.reviews.id", "context.reviews.owner_id", "context.reviews.version_id"]),
    ForeignKeyConstraint(["revision_id", "owner_id", "version_id"], ["context.revisions.id", "context.revisions.owner_id", "context.revisions.version_id"]),
    CheckConstraint("status IN ('idle', 'running', 'interrupted')"),
)
commands = Table(
    "commands",
    metadata,
    *identity(),
    Column("conversation_id", String(36), nullable=False),
    Column("request_hash", String(64), nullable=False),
    Column("body", JSONB, nullable=False),
    Column("status", String(20), nullable=False, server_default="accepted"),
    Column("result", JSONB),
    stamp(),
    parent("conversation_id", "conversations"),
    UniqueConstraint("id", "owner_id", "conversation_id"),
)
messages = Table(
    "messages",
    metadata,
    *identity(),
    Column("conversation_id", String(36), nullable=False),
    Column("operation_id", String(36), nullable=False),
    Column("role", String(20), nullable=False),
    Column("text", Text, nullable=False),
    Column("references", JSONB, nullable=False, server_default="[]"),
    stamp(),
    parent("conversation_id", "conversations"),
    UniqueConstraint("conversation_id", "operation_id", "role"),
)
questions = Table(
    "questions",
    metadata,
    *identity(),
    Column("conversation_id", String(36), nullable=False),
    Column("revision_id", String(36), nullable=False),
    Column("target", Text, nullable=False),
    Column("body", JSONB, nullable=False),
    Column("answer", JSONB),
    stamp(),
    parent("conversation_id", "conversations"),
    UniqueConstraint("conversation_id", "revision_id", "target"),
)
selections = Table(
    "artifact_selections",
    metadata,
    Column("conversation_id", String(36), primary_key=True),
    Column("artifact_id", String(36), primary_key=True),
    Column("owner_id", String(36), nullable=False),
    Column("operation_id", String(36), nullable=False),
    stamp(),
    parent("conversation_id", "conversations"),
    parent("artifact_id", "artifacts"),
)
