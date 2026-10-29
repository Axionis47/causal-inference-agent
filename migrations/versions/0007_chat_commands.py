"""Pinned chat selections, idempotent commands, messages and structured questions."""

from alembic import op
from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, ForeignKeyConstraint, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB

revision = "0007_chat_commands"
down_revision = "0006_reviewed_context"
branch_labels = None
depends_on = None


def parent():
    return ForeignKeyConstraint(["conversation_id", "owner_id"], ["chat.conversations.id", "chat.conversations.owner_id"])


def upgrade():
    op.create_table(
        "state",
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
        Column("error", Text),
        parent(),
        ForeignKeyConstraint(
            ["conversation_id", "owner_id", "version_id"], ["chat.conversations.id", "chat.conversations.owner_id", "chat.conversations.version_id"]
        ),
        ForeignKeyConstraint(["review_id", "owner_id", "version_id"], ["context.reviews.id", "context.reviews.owner_id", "context.reviews.version_id"]),
        ForeignKeyConstraint(["revision_id", "owner_id", "version_id"], ["context.revisions.id", "context.revisions.owner_id", "context.revisions.version_id"]),
        CheckConstraint("status IN ('idle', 'running', 'interrupted')"),
        schema="chat",
    )
    op.create_table(
        "commands",
        Column("id", String(36), primary_key=True),
        Column("owner_id", String(36), nullable=False),
        Column("conversation_id", String(36), nullable=False),
        Column("request_hash", String(64), nullable=False),
        Column("body", JSONB, nullable=False),
        Column("status", String(20), nullable=False, server_default="accepted"),
        Column("result", JSONB),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        parent(),
        UniqueConstraint("id", "owner_id"),
        UniqueConstraint("id", "owner_id", "conversation_id"),
        schema="chat",
    )
    op.create_table(
        "messages",
        Column("id", String(36), primary_key=True),
        Column("owner_id", String(36), nullable=False),
        Column("conversation_id", String(36), nullable=False),
        Column("operation_id", String(36), nullable=False),
        Column("role", String(20), nullable=False),
        Column("text", Text, nullable=False),
        Column("references", JSONB, nullable=False, server_default="[]"),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        parent(),
        UniqueConstraint("id", "owner_id"),
        UniqueConstraint("conversation_id", "operation_id", "role"),
        schema="chat",
    )
    op.create_table(
        "questions",
        Column("id", String(36), primary_key=True),
        Column("owner_id", String(36), nullable=False),
        Column("conversation_id", String(36), nullable=False),
        Column("revision_id", String(36), nullable=False),
        Column("target", Text, nullable=False),
        Column("body", JSONB, nullable=False),
        Column("answer", JSONB),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        parent(),
        UniqueConstraint("id", "owner_id"),
        UniqueConstraint("conversation_id", "revision_id", "target"),
        schema="chat",
    )
    op.create_index("chat_messages_page", "messages", ["conversation_id", "created_at", "id"], schema="chat")


def downgrade():
    for name in ("questions", "messages", "commands", "state"):
        op.drop_table(name, schema="chat")
