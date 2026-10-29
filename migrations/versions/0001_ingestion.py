"""Initial owned ingestion schema. Independent of future application metadata."""

from alembic import op

revision = "0001_ingestion"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE SCHEMA ingestion")
    op.execute("""
CREATE TABLE ingestion.users (
	id VARCHAR(36) NOT NULL,
	PRIMARY KEY (id)
)

""")
    op.execute("""
CREATE TABLE ingestion.operations (
	id VARCHAR(36) NOT NULL,
	owner_id VARCHAR(36) NOT NULL,
	idempotency_key VARCHAR(128) NOT NULL,
	fingerprint VARCHAR(64) NOT NULL,
	object_key TEXT NOT NULL,
	status VARCHAR(16) NOT NULL,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
	PRIMARY KEY (id),
	UNIQUE (id, owner_id),
	FOREIGN KEY(owner_id) REFERENCES ingestion.users (id),
	UNIQUE (owner_id, idempotency_key),
	CHECK (status IN ('pending', 'ready', 'failed')),
	UNIQUE (object_key)
)

""")
    op.execute("""
CREATE TABLE ingestion.datasets (
	id VARCHAR(36) NOT NULL,
	owner_id VARCHAR(36) NOT NULL,
	title VARCHAR(200) NOT NULL,
	operation_id VARCHAR(36) NOT NULL,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
	PRIMARY KEY (id),
	UNIQUE (id, owner_id),
	FOREIGN KEY(owner_id) REFERENCES ingestion.users (id),
	FOREIGN KEY(operation_id, owner_id) REFERENCES ingestion.operations (id, owner_id),
	UNIQUE (operation_id)
)

""")
    op.execute("""
CREATE TABLE ingestion.files (
	id VARCHAR(36) NOT NULL,
	owner_id VARCHAR(36) NOT NULL,
	operation_id VARCHAR(36) NOT NULL,
	object_key TEXT NOT NULL,
	filename TEXT NOT NULL,
	sha256 VARCHAR(64) NOT NULL,
	size_bytes BIGINT NOT NULL,
	PRIMARY KEY (id),
	UNIQUE (id, owner_id),
	FOREIGN KEY(owner_id) REFERENCES ingestion.users (id),
	FOREIGN KEY(operation_id, owner_id) REFERENCES ingestion.operations (id, owner_id),
	CHECK (size_bytes > 0),
	UNIQUE (operation_id),
	UNIQUE (object_key)
)

""")
    op.execute("""
CREATE TABLE ingestion.versions (
	id VARCHAR(36) NOT NULL,
	owner_id VARCHAR(36) NOT NULL,
	dataset_id VARCHAR(36) NOT NULL,
	file_id VARCHAR(36) NOT NULL,
	number INTEGER NOT NULL,
	PRIMARY KEY (id),
	UNIQUE (id, owner_id),
	FOREIGN KEY(owner_id) REFERENCES ingestion.users (id),
	FOREIGN KEY(dataset_id, owner_id) REFERENCES ingestion.datasets (id, owner_id),
	FOREIGN KEY(file_id, owner_id) REFERENCES ingestion.files (id, owner_id),
	UNIQUE (dataset_id, number),
	CHECK (number = 0)
)

""")
    op.execute("""
CREATE TABLE ingestion.contexts (
	id VARCHAR(36) NOT NULL,
	owner_id VARCHAR(36) NOT NULL,
	version_id VARCHAR(36) NOT NULL,
	text TEXT NOT NULL,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
	PRIMARY KEY (id),
	UNIQUE (id, owner_id),
	FOREIGN KEY(owner_id) REFERENCES ingestion.users (id),
	FOREIGN KEY(version_id, owner_id) REFERENCES ingestion.versions (id, owner_id),
	UNIQUE (version_id)
)

""")
    op.execute(
        "CREATE FUNCTION ingestion.immutable() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'Original ingestion records are immutable'; END $$"
    )
    op.execute("CREATE TRIGGER immutable BEFORE UPDATE ON ingestion.files FOR EACH ROW EXECUTE FUNCTION ingestion.immutable()")
    op.execute("CREATE TRIGGER immutable BEFORE UPDATE ON ingestion.versions FOR EACH ROW EXECUTE FUNCTION ingestion.immutable()")
    op.execute("CREATE TRIGGER immutable BEFORE UPDATE ON ingestion.contexts FOR EACH ROW EXECUTE FUNCTION ingestion.immutable()")


def downgrade():
    op.execute("DROP SCHEMA ingestion CASCADE")
