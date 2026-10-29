"""Relational ownership constraints for ingestion, mirrored by the initial migration."""

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)

metadata = MetaData(schema="ingestion")


def identity():
    return [Column("id", String(36), primary_key=True), Column("owner_id", String(36), nullable=False), UniqueConstraint("id", "owner_id")]


def owned():
    return ForeignKeyConstraint(["owner_id"], ["ingestion.users.id"])


def parent(column, table):
    return ForeignKeyConstraint([column, "owner_id"], [f"ingestion.{table}.id", f"ingestion.{table}.owner_id"])


users = Table("users", metadata, Column("id", String(36), primary_key=True))
operations = Table(
    "operations",
    metadata,
    *identity(),
    owned(),
    Column("idempotency_key", String(128), nullable=False),
    Column("fingerprint", String(64), nullable=False),
    Column("object_key", Text, nullable=False, unique=True),
    Column("status", String(16), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("owner_id", "idempotency_key"),
    CheckConstraint("status IN ('pending', 'ready', 'failed')"),
)
files = Table(
    "files",
    metadata,
    *identity(),
    owned(),
    Column("operation_id", String(36), nullable=False, unique=True),
    parent("operation_id", "operations"),
    Column("object_key", Text, nullable=False, unique=True),
    Column("filename", Text, nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    CheckConstraint("size_bytes > 0"),
)
datasets = Table(
    "datasets",
    metadata,
    *identity(),
    owned(),
    Column("title", String(200), nullable=False),
    Column("operation_id", String(36), nullable=False, unique=True),
    parent("operation_id", "operations"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
versions = Table(
    "versions",
    metadata,
    *identity(),
    owned(),
    Column("dataset_id", String(36), nullable=False),
    parent("dataset_id", "datasets"),
    Column("file_id", String(36), nullable=False),
    parent("file_id", "files"),
    Column("number", Integer, nullable=False),
    UniqueConstraint("dataset_id", "number"),
    UniqueConstraint("id", "owner_id", "dataset_id", name="versions_identity_dataset"),
    CheckConstraint("number = 0"),
)
contexts = Table(
    "contexts",
    metadata,
    *identity(),
    owned(),
    Column("version_id", String(36), nullable=False, unique=True),
    UniqueConstraint("id", "owner_id", "version_id", name="contexts_identity_version"),
    parent("version_id", "versions"),
    Column("text", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

Index("datasets_owner_created_id", datasets.c.owner_id, datasets.c.created_at.desc(), datasets.c.id.desc())
