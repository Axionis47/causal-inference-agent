"""Retain completed A2A task identities beside their application attempt."""

from alembic import op
from sqlalchemy import Column, Text

revision = "0011_external_tasks"
down_revision = "0010_chat_readiness"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("attempts", Column("external_task_id", Text), schema="chat")


def downgrade():
    op.drop_column("attempts", "external_task_id", schema="chat")
