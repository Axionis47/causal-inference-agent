"""Context identity is relational; bounded semantic statements are JSONB."""

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKeyConstraint, Integer, MetaData, String, Table, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB

from causal_agent.server.ingestion.schema import contexts, versions
from causal_agent.server.profiling.schema import columns

metadata = MetaData(schema="context")
versions.to_metadata(metadata, schema="ingestion")
contexts.to_metadata(metadata, schema="ingestion")
columns.to_metadata(metadata, schema="profiling")

packs = Table(
    "packs",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("owner_id", String(36), nullable=False),
    Column("version_id", String(36), nullable=False),
    Column("source_id", String(36)),
    Column("revision", Integer, nullable=False),
    Column("identity", String(64), nullable=False),
    Column("source_checksum", String(64), nullable=False),
    Column("csv_checksum", String(64), nullable=False),
    Column("column_index_fingerprint", String(64), nullable=False),
    Column("schema_version", String(40), nullable=False),
    Column("extractor_version", String(40), nullable=False),
    Column("model", String(200), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("dataset", JSONB, nullable=False),
    Column("concepts", JSONB, nullable=False),
    Column("assessment", JSONB, nullable=False),
    Column("model_calls", Integer, nullable=False),
    UniqueConstraint("id", "owner_id", "version_id"),
    UniqueConstraint("version_id", "revision"),
    UniqueConstraint("version_id", "identity"),
    ForeignKeyConstraint(["version_id", "owner_id"], ["ingestion.versions.id", "ingestion.versions.owner_id"]),
    ForeignKeyConstraint(["source_id", "owner_id", "version_id"], ["ingestion.contexts.id", "ingestion.contexts.owner_id", "ingestion.contexts.version_id"]),
    CheckConstraint("revision > 0"),
    CheckConstraint("model_calls BETWEEN 0 AND 4"),
)


def parent():
    return ForeignKeyConstraint(["pack_id", "owner_id", "version_id"], ["context.packs.id", "context.packs.owner_id", "context.packs.version_id"])


column_contexts = Table(
    "column_contexts",
    metadata,
    Column("pack_id", String(36), primary_key=True),
    Column("column_id", String(36), primary_key=True),
    Column("owner_id", String(36), nullable=False),
    Column("version_id", String(36), nullable=False),
    Column("statements", JSONB, nullable=False),
    parent(),
    ForeignKeyConstraint(["column_id", "owner_id", "version_id"], ["profiling.columns.id", "profiling.columns.owner_id", "profiling.columns.version_id"]),
)
relationships = Table(
    "relationships",
    metadata,
    Column("pack_id", String(36), primary_key=True),
    Column("id", String(80), primary_key=True),
    Column("owner_id", String(36), nullable=False),
    Column("version_id", String(36), nullable=False),
    Column("result", JSONB, nullable=False),
    parent(),
)

# One context revision chain, beginning at extraction. Tables above are immutable extraction provenance only.
revisions = Table(
    "revisions",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("owner_id", String(36), nullable=False),
    Column("version_id", String(36), nullable=False),
    Column("source_pack_id", String(36), nullable=False),
    Column("parent_id", String(36)),
    Column("number", Integer, nullable=False),
    Column("knowledge", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("id", "owner_id", "version_id"),
    UniqueConstraint("version_id", "number"),
    ForeignKeyConstraint(["source_pack_id", "owner_id", "version_id"], ["context.packs.id", "context.packs.owner_id", "context.packs.version_id"]),
    ForeignKeyConstraint(["parent_id", "owner_id", "version_id"], ["context.revisions.id", "context.revisions.owner_id", "context.revisions.version_id"]),
    CheckConstraint("number > 0"),
)
reviews = Table(
    "reviews",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("owner_id", String(36), nullable=False),
    Column("version_id", String(36), nullable=False),
    Column("question", Text, nullable=False),
    Column("starting_revision_id", String(36), nullable=False),
    Column("completed_revision_id", String(36)),
    Column("generation", Integer, nullable=False, server_default="1"),
    Column("edits", JSONB, nullable=False, server_default="[]"),
    Column("dispositions", JSONB, nullable=False, server_default="[]"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("id", "owner_id", "version_id"),
    *[
        ForeignKeyConstraint([key, "owner_id", "version_id"], ["context.revisions.id", "context.revisions.owner_id", "context.revisions.version_id"])
        for key in ("starting_revision_id", "completed_revision_id")
    ],
)
receipts = Table(
    "receipts",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("owner_id", String(36), nullable=False),
    Column("version_id", String(36), nullable=False),
    Column("request_hash", String(64), nullable=False),
    Column("revision_id", String(36), nullable=False),
    Column("source", JSONB, nullable=False),
    Column("edits", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    ForeignKeyConstraint(["revision_id", "owner_id", "version_id"], ["context.revisions.id", "context.revisions.owner_id", "context.revisions.version_id"]),
)
