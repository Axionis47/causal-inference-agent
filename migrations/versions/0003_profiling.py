"""Owned dataset and column profiling; originals remain unchanged."""

from alembic import op

revision = "0003_profiling"
down_revision = "0002_library_index"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE SCHEMA profiling")
    op.execute(
        "\nCREATE TABLE profiling.columns (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(36) NOT NULL, \n\tversion_id VARCHAR(36) NOT NULL, \n\tposition INTEGER NOT NULL, \n\tname TEXT NOT NULL, \n\tkey TEXT NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (id, owner_id, version_id), \n\tFOREIGN KEY(version_id, owner_id) REFERENCES ingestion.versions (id, owner_id), \n\tUNIQUE (version_id, position), \n\tUNIQUE (version_id, key), \n\tCHECK (position >= 0)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE profiling.profiles (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(36) NOT NULL, \n\tversion_id VARCHAR(36) NOT NULL, \n\tsource_checksum VARCHAR(64) NOT NULL, \n\tprofiler_version VARCHAR(40) NOT NULL, \n\tsettings_fingerprint VARCHAR(64) NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tupdated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tstatus VARCHAR(16) NOT NULL, \n\tdataset JSONB NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (id, owner_id, version_id), \n\tFOREIGN KEY(version_id, owner_id) REFERENCES ingestion.versions (id, owner_id), \n\tUNIQUE (version_id, source_checksum, profiler_version, settings_fingerprint), \n\tCHECK (status IN ('complete', 'partial', 'failed'))\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE profiling.column_results (\n\tprofile_id VARCHAR(36) NOT NULL, \n\tcolumn_id VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(36) NOT NULL, \n\tversion_id VARCHAR(36) NOT NULL, \n\tstatus VARCHAR(16) NOT NULL, \n\tresult JSONB NOT NULL, \n\tPRIMARY KEY (profile_id, column_id), \n\tFOREIGN KEY(profile_id, owner_id, version_id) REFERENCES profiling.profiles (id, owner_id, version_id), \n\tFOREIGN KEY(column_id, owner_id, version_id) REFERENCES profiling.columns (id, owner_id, version_id), \n\tCHECK (status IN ('complete', 'partial', 'failed'))\n)\n\n"
    )


def downgrade():
    op.execute("DROP TABLE profiling.column_results")
    op.execute("DROP TABLE profiling.profiles")
    op.execute("DROP TABLE profiling.columns")
    op.execute("DROP SCHEMA profiling")
