-- Intake catalogue schema and narrow retrieval views (PRD-001 §9.3; T-008; D-029).
CREATE SCHEMA catalog;

CREATE TABLE catalog.runs (
    analysis_id                text PRIMARY KEY,
    stage_run_id               text NOT NULL REFERENCES causal.stage_runs (stage_run_id),
    question_artifact_id       text NOT NULL REFERENCES causal.artifacts (artifact_id),
    dataset_id                 text,
    intake_outcome_artifact_id text REFERENCES causal.artifacts (artifact_id),
    intake_status              text CHECK (intake_status IN
        ('usable', 'partial', 'refused', 'failed', 'failed_observability')),
    idempotency_key            text NOT NULL UNIQUE,
    submission_hash            text NOT NULL CHECK (submission_hash ~ '^[a-f0-9]{64}$'),
    created_at_utc             timestamptz NOT NULL
);

CREATE TABLE catalog.datasets (
    dataset_id                  text PRIMARY KEY,
    provider                    text NOT NULL CHECK (provider = 'kaggle'),
    owner                       text NOT NULL,
    slug                        text NOT NULL,
    version                     text NOT NULL,
    provider_status             text NOT NULL,
    capture_artifact_id         text NOT NULL REFERENCES causal.artifacts (artifact_id),
    source_manifest_artifact_id text REFERENCES causal.artifacts (artifact_id)
);

CREATE TABLE catalog.resources (
    resource_id   text PRIMARY KEY,
    dataset_id    text NOT NULL REFERENCES catalog.datasets (dataset_id),
    kind          text NOT NULL CHECK (kind IN
        ('api_capture', 'archive', 'table', 'document', 'metadata', 'other')),
    logical_name  text NOT NULL,
    object_sha256 text NOT NULL CHECK (object_sha256 ~ '^[a-f0-9]{64}$'),
    object_key    text NOT NULL,
    media_type    text,
    byte_size     bigint NOT NULL CHECK (byte_size >= 0),
    parse_status  text NOT NULL CHECK (parse_status IN
        ('parsed', 'excluded', 'unreadable', 'unsafe', 'failed')),
    reason        text CHECK (parse_status = 'parsed' OR reason IS NOT NULL),
    UNIQUE (dataset_id, logical_name)
);

CREATE TABLE catalog.source_field_index (
    dataset_id               text NOT NULL REFERENCES catalog.datasets (dataset_id),
    capture_artifact_id      text NOT NULL REFERENCES causal.artifacts (artifact_id),
    semantic_map_artifact_id text REFERENCES causal.artifacts (artifact_id),
    scope_kind               text NOT NULL CHECK (scope_kind IN ('dataset', 'table', 'column')),
    table_name               text,
    column_name              text,
    field_or_slot_name       text NOT NULL,
    context_class            text NOT NULL CHECK (context_class IN
        ('semantic', 'structural', 'measured', 'provenance',
         'operational', 'popularity', 'withheld')),
    status                   text NOT NULL CHECK (status IN
        ('evidenced', 'hinted', 'hypothesis', 'empty', 'not_offered',
         'fetch_failed', 'unreadable', 'withheld', 'not_applicable')),
    evidence_count           integer NOT NULL CHECK (evidence_count >= 0),
    json_pointer             text NOT NULL,
    UNIQUE NULLS NOT DISTINCT
        (dataset_id, scope_kind, table_name, column_name, field_or_slot_name)
);

-- The five narrow views; deliberately no all_context view (PRD-001 §9.3).
CREATE VIEW catalog.semantic_available AS
    SELECT dataset_id, scope_kind, table_name, column_name, field_or_slot_name,
           status, evidence_count, json_pointer, capture_artifact_id,
           semantic_map_artifact_id
    FROM catalog.source_field_index
    WHERE context_class = 'semantic'
      AND status IN ('evidenced', 'hinted', 'hypothesis');

CREATE VIEW catalog.semantic_missing AS
    SELECT dataset_id, scope_kind, table_name, column_name, field_or_slot_name,
           status
    FROM catalog.source_field_index
    WHERE context_class = 'semantic'
      AND status NOT IN ('evidenced', 'hinted', 'hypothesis');

CREATE VIEW catalog.structural_manifest AS
    SELECT dataset_id, scope_kind, table_name, column_name, field_or_slot_name,
           status, json_pointer
    FROM catalog.source_field_index
    WHERE context_class = 'structural';

CREATE VIEW catalog.measured_fact_manifest AS
    SELECT dataset_id, scope_kind, table_name, column_name, field_or_slot_name,
           status, json_pointer
    FROM catalog.source_field_index
    WHERE context_class = 'measured';

CREATE VIEW catalog.provenance_manifest AS
    SELECT dataset_id, scope_kind, table_name, column_name, field_or_slot_name,
           status, json_pointer
    FROM catalog.source_field_index
    WHERE context_class = 'provenance';
