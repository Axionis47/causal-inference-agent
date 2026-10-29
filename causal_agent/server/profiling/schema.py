"""Owned profile records, with relational identity outside bounded JSON measurements."""

from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKeyConstraint, Integer, MetaData, String, Table, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB

from causal_agent.server.ingestion.schema import versions

metadata = MetaData(schema="profiling")
versions.to_metadata(metadata, schema="ingestion")


def identity():
    return [
        Column("id", String(36), primary_key=True),
        Column("owner_id", String(36), nullable=False),
        Column("version_id", String(36), nullable=False),
        UniqueConstraint("id", "owner_id", "version_id"),
        ForeignKeyConstraint(["version_id", "owner_id"], ["ingestion.versions.id", "ingestion.versions.owner_id"]),
    ]


columns = Table(
    "columns",
    metadata,
    *identity(),
    Column("position", Integer, nullable=False),
    Column("name", Text, nullable=False),
    Column("key", Text, nullable=False),
    UniqueConstraint("version_id", "position"),
    UniqueConstraint("version_id", "key"),
    CheckConstraint("position >= 0"),
)
profiles = Table(
    "profiles",
    metadata,
    *identity(),
    Column("source_checksum", String(64), nullable=False),
    Column("profiler_version", String(40), nullable=False),
    Column("settings_fingerprint", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("status", String(16), nullable=False),
    Column("dataset", JSONB, nullable=False),
    UniqueConstraint("version_id", "source_checksum", "profiler_version", "settings_fingerprint"),
    CheckConstraint("status IN ('complete', 'partial', 'failed')"),
)
column_results = Table(
    "column_results",
    metadata,
    Column("profile_id", String(36), primary_key=True),
    Column("column_id", String(36), primary_key=True),
    Column("owner_id", String(36), nullable=False),
    Column("version_id", String(36), nullable=False),
    Column("status", String(16), nullable=False),
    Column("result", JSONB, nullable=False),
    ForeignKeyConstraint(["profile_id", "owner_id", "version_id"], ["profiling.profiles.id", "profiling.profiles.owner_id", "profiling.profiles.version_id"]),
    ForeignKeyConstraint(["column_id", "owner_id", "version_id"], ["profiling.columns.id", "profiling.columns.owner_id", "profiling.columns.version_id"]),
    CheckConstraint("status IN ('complete', 'partial', 'failed')"),
)
