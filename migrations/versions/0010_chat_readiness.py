"""Persist the question frame and small endpoint-readiness projection."""

from alembic import op
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB

revision = "0010_chat_readiness"
down_revision = "0009_revision_immutability"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("state", Column("frame", JSONB), schema="chat")
    op.add_column("state", Column("readiness", JSONB, nullable=False, server_default='{"status":"unprepared","missing":0}'), schema="chat")


def downgrade():
    op.drop_column("state", "readiness", schema="chat")
    op.drop_column("state", "frame", schema="chat")
