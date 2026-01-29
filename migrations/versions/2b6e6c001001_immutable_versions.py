"""Protect saved version identity and metadata against in-place updates."""

from alembic import op

revision = "2b6e6c001001"
down_revision = "e710795a3a76"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE FUNCTION reject_version_update() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          RAISE EXCEPTION 'Dataset versions are immutable; create a new version instead';
        END;
        $$;
        CREATE TRIGGER dataset_version_immutable
        BEFORE UPDATE ON dataset_versions FOR EACH ROW EXECUTE FUNCTION reject_version_update();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER dataset_version_immutable ON dataset_versions")
    op.execute("DROP FUNCTION reject_version_update()")
