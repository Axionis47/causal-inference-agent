"""Owned conversations and atomic exploration publication."""

from alembic import op

revision = "0005_conversations"
down_revision = "0004_context"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE ingestion.versions ADD CONSTRAINT versions_identity_dataset UNIQUE (id, owner_id, dataset_id)")
    op.execute("CREATE SCHEMA chat")
    op.execute(
        "\nCREATE TABLE chat.conversations (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(36) NOT NULL, \n\tdataset_id VARCHAR(36) NOT NULL, \n\tversion_id VARCHAR(36) NOT NULL, \n\ttitle TEXT NOT NULL, \n\tcontext TEXT NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (id, owner_id), \n\tFOREIGN KEY(dataset_id, owner_id) REFERENCES ingestion.datasets (id, owner_id), \n\tFOREIGN KEY(version_id, owner_id, dataset_id) REFERENCES ingestion.versions (id, owner_id, dataset_id), \n\tUNIQUE (id, owner_id, version_id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE chat.exploration_requests (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(36) NOT NULL, \n\tconversation_id VARCHAR(36) NOT NULL, \n\tversion_id VARCHAR(36) NOT NULL, \n\tsource_sha256 VARCHAR(64) NOT NULL, \n\tquestion TEXT NOT NULL, \n\tcontext TEXT NOT NULL, \n\tparent_refs JSONB NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (id, owner_id), \n\tFOREIGN KEY(conversation_id, owner_id, version_id) REFERENCES chat.conversations (id, owner_id, version_id), \n\tUNIQUE (id, owner_id, conversation_id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE chat.attempts (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(36) NOT NULL, \n\trequest_id VARCHAR(36) NOT NULL, \n\tconversation_id VARCHAR(36) NOT NULL, \n\tstatus VARCHAR(20) NOT NULL, \n\toutput_manifest JSONB DEFAULT '[]' NOT NULL, \n\tdetail TEXT DEFAULT '' NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tended_at TIMESTAMP WITH TIME ZONE, \n\tPRIMARY KEY (id), \n\tUNIQUE (id, owner_id), \n\tFOREIGN KEY(request_id, owner_id, conversation_id) REFERENCES chat.exploration_requests (id, owner_id, conversation_id), \n\tUNIQUE (id, owner_id, conversation_id), \n\tCHECK (status IN ('running', 'publishing', 'completed', 'failed', 'needs_input'))\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE chat.exploration_results (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(36) NOT NULL, \n\tattempt_id VARCHAR(36) NOT NULL, \n\tconversation_id VARCHAR(36) NOT NULL, \n\trecord JSONB NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (id, owner_id), \n\tFOREIGN KEY(attempt_id, owner_id, conversation_id) REFERENCES chat.attempts (id, owner_id, conversation_id), \n\tUNIQUE (id, owner_id, conversation_id), \n\tUNIQUE (attempt_id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE chat.artifacts (\n\tid VARCHAR(36) NOT NULL, \n\towner_id VARCHAR(36) NOT NULL, \n\tconversation_id VARCHAR(36) NOT NULL, \n\tresult_id VARCHAR(36) NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (id, owner_id), \n\tFOREIGN KEY(result_id, owner_id, conversation_id) REFERENCES chat.exploration_results (id, owner_id, conversation_id), \n\tUNIQUE (id, owner_id, conversation_id)\n)\n\n"
    )
    op.execute(
        "\nCREATE TABLE chat.artifact_files (\n\tartifact_id VARCHAR(36) NOT NULL, \n\tname VARCHAR(120) NOT NULL, \n\towner_id VARCHAR(36) NOT NULL, \n\tconversation_id VARCHAR(36) NOT NULL, \n\tobject_key TEXT NOT NULL, \n\tsha256 VARCHAR(64) NOT NULL, \n\tsize_bytes BIGINT NOT NULL, \n\tPRIMARY KEY (artifact_id, name), \n\tFOREIGN KEY(artifact_id, owner_id, conversation_id) REFERENCES chat.artifacts (id, owner_id, conversation_id), \n\tCHECK (size_bytes >= 0), \n\tUNIQUE (object_key)\n)\n\n"
    )


def downgrade():
    op.execute("DROP TABLE chat.artifact_files")
    op.execute("DROP TABLE chat.artifacts")
    op.execute("DROP TABLE chat.exploration_results")
    op.execute("DROP TABLE chat.attempts")
    op.execute("DROP TABLE chat.exploration_requests")
    op.execute("DROP TABLE chat.conversations")
    op.execute("DROP SCHEMA chat")
    op.execute("ALTER TABLE ingestion.versions DROP CONSTRAINT versions_identity_dataset")
