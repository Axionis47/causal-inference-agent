"""Immutable reviewed context and durable pre-chat reviews."""

from alembic import op
from sqlalchemy import Column, DateTime, ForeignKeyConstraint, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB

revision = "0006_reviewed_context"
down_revision = "0005_conversations"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "revisions",
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
        schema="context",
    )
    op.create_check_constraint("reviewed_revision_positive", "revisions", "number > 0", schema="context")
    op.create_table(
        "reviews",
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
        schema="context",
    )
    op.create_table(
        "receipts",
        Column("id", String(36), primary_key=True),
        Column("owner_id", String(36), nullable=False),
        Column("version_id", String(36), nullable=False),
        Column("request_hash", String(64), nullable=False),
        Column("revision_id", String(36), nullable=False),
        Column("source", JSONB, nullable=False),
        Column("edits", JSONB, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        ForeignKeyConstraint(["revision_id", "owner_id", "version_id"], ["context.revisions.id", "context.revisions.owner_id", "context.revisions.version_id"]),
        schema="context",
    )


def downgrade():
    for name in ("receipts", "reviews", "revisions"):
        op.drop_table(name, schema="context")
