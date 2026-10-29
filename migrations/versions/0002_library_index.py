"""Index owned library pagination without changing ingested records."""

from alembic import op

revision = "0002_library_index"
down_revision = "0001_ingestion"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE INDEX datasets_owner_created_id ON ingestion.datasets (owner_id, created_at DESC, id DESC)")


def downgrade():
    op.drop_index("datasets_owner_created_id", table_name="datasets", schema="ingestion")
