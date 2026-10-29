"""Keep reviewed context, receipts and published messages immutable."""

from alembic import op

revision = "0009_revision_immutability"
down_revision = "0008_artifact_selections"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "CREATE FUNCTION context.immutable_record() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'Published records are immutable'; END $$"
    )
    for table in ("context.revisions", "context.receipts", "chat.messages", "chat.exploration_results"):
        op.execute(f"CREATE TRIGGER immutable BEFORE UPDATE ON {table} FOR EACH ROW EXECUTE FUNCTION context.immutable_record()")


def downgrade():
    for table in ("context.revisions", "context.receipts", "chat.messages", "chat.exploration_results"):
        op.execute(f"DROP TRIGGER immutable ON {table}")
    op.execute("DROP FUNCTION context.immutable_record()")
