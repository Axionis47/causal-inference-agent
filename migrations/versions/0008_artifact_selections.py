"""Explicit same-dataset artifact reuse with preserved producer identity."""

from alembic import op
from sqlalchemy import Column, DateTime, ForeignKeyConstraint, String, func

revision = "0008_artifact_selections"
down_revision = "0007_chat_commands"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "artifact_selections",
        Column("conversation_id", String(36), primary_key=True),
        Column("artifact_id", String(36), primary_key=True),
        Column("owner_id", String(36), nullable=False),
        Column("operation_id", String(36), nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
        ForeignKeyConstraint(["conversation_id", "owner_id"], ["chat.conversations.id", "chat.conversations.owner_id"]),
        ForeignKeyConstraint(["artifact_id", "owner_id"], ["chat.artifacts.id", "chat.artifacts.owner_id"]),
        schema="chat",
    )


def downgrade():
    op.drop_table("artifact_selections", schema="chat")
