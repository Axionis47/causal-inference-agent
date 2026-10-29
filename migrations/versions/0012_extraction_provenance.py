"""Extraction outputs are immutable provenance, never another current context."""

from alembic import op

revision = "0012_extraction_provenance"
down_revision = "0011_external_tasks"
branch_labels = None
depends_on = None


def upgrade():
    for table in ("context.packs", "context.column_contexts", "context.relationships"):
        op.execute(f"CREATE TRIGGER immutable BEFORE UPDATE ON {table} FOR EACH ROW EXECUTE FUNCTION context.immutable_record()")


def downgrade():
    for table in ("context.packs", "context.column_contexts", "context.relationships"):
        op.execute(f"DROP TRIGGER immutable ON {table}")
