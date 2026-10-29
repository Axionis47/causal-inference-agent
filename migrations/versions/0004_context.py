"""Organized semantic context; preserves original inputs and profiles."""

from alembic import op

revision = "0004_context"
down_revision = "0003_profiling"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE ingestion.contexts ADD CONSTRAINT contexts_identity_version UNIQUE (id, owner_id, version_id)")
    op.execute("CREATE SCHEMA context")
    op.execute(
        "\nCREATE TABLE context.packs (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(36) NOT NULL, \n\tversion_id VARCHAR(36) NOT NULL, \n\tsource_id VARCHAR(36), \n\trevision INTEGER NOT NULL, \n\tidentity VARCHAR(64) NOT NULL, \n\tsource_checksum VARCHAR(64) NOT NULL, \n\tcsv_checksum VARCHAR(64) NOT NULL, \n\tcolumn_index_fingerprint VARCHAR(64) NOT NULL, \n\tschema_version VARCHAR(40) NOT NULL, \n\textractor_version VARCHAR(40) NOT NULL, \n\tmodel VARCHAR(200) NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tdataset JSONB NOT NULL, \n\tconcepts JSONB NOT NULL, \n\tassessment JSONB NOT NULL, \n\tmodel_calls INTEGER NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (id, owner_id, version_id), \n\tUNIQUE (version_id, revision), \n\tUNIQUE (version_id, identity), \n\tFOREIGN KEY(version_id, owner_id) REFERENCES ingestion.versions (id, owner_id), \n\tFOREIGN KEY(source_id, owner_id, version_id) REFERENCES ingestion.contexts (id, owner_id, version_id), \n\tCHECK (revision > 0), \n\tCHECK (model_calls BETWEEN 0 AND 4)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE context.column_contexts (\n\tpack_id VARCHAR(36) NOT NULL, \n\tcolumn_id VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(36) NOT NULL, \n\tversion_id VARCHAR(36) NOT NULL, \n\tstatements JSONB NOT NULL, \n\tPRIMARY KEY (pack_id, column_id), \n\tFOREIGN KEY(pack_id, owner_id, version_id) REFERENCES context.packs (id, owner_id, version_id), \n\tFOREIGN KEY(column_id, owner_id, version_id) REFERENCES profiling.columns (id, owner_id, version_id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE context.relationships (\n\tpack_id VARCHAR(36) NOT NULL, \n\tid VARCHAR(80) NOT NULL, \n\towner_id VARCHAR(36) NOT NULL, \n\tversion_id VARCHAR(36) NOT NULL, \n\tresult JSONB NOT NULL, \n\tPRIMARY KEY (pack_id, id), \n\tFOREIGN KEY(pack_id, owner_id, version_id) REFERENCES context.packs (id, owner_id, version_id)\n)\n\n"
    )
    op.execute("CREATE INDEX context_relationship_participants ON context.relationships USING gin ((result -> 'participants'))")


def downgrade():
    op.execute("DROP TABLE context.relationships")
    op.execute("DROP TABLE context.column_contexts")
    op.execute("DROP TABLE context.packs")
    op.execute("DROP SCHEMA context")
    op.execute("ALTER TABLE ingestion.contexts DROP CONSTRAINT contexts_identity_version")
