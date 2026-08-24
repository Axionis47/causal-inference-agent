-- Shared product-state schema (SYSTEM-CONTRACT §8.1, §8.2; task T-005).
CREATE SCHEMA IF NOT EXISTS causal;

CREATE TABLE causal.stage_runs (
    stage_run_id   text PRIMARY KEY,
    analysis_id    text NOT NULL,
    stage          text NOT NULL CHECK (stage IN
        ('intake', 'design', 'preparation', 'estimation', 'presentation', 'system')),
    run_state      text NOT NULL CHECK (run_state IN
        ('created', 'tracing_preflight', 'running', 'waiting_for_user',
         'committing', 'completed', 'failed_observability', 'failed')),
    created_at_utc timestamptz NOT NULL,
    updated_at_utc timestamptz NOT NULL
);

CREATE INDEX stage_runs_analysis_idx ON causal.stage_runs (analysis_id);

CREATE TABLE causal.artifacts (
    artifact_id        text PRIMARY KEY,
    artifact_type      text NOT NULL,
    schema_version     text NOT NULL,
    content_hash       text NOT NULL CHECK (content_hash ~ '^[a-f0-9]{64}$'),
    analysis_id        text NOT NULL,
    stage_run_id       text NOT NULL REFERENCES causal.stage_runs (stage_run_id),
    producer_component text NOT NULL,
    producer_version   text NOT NULL,
    sensitivity_class  text NOT NULL CHECK (sensitivity_class IN
        ('public', 'internal', 'restricted', 'secret_reference')),
    created_at_utc     timestamptz NOT NULL,
    payload_locator    text NOT NULL
);

CREATE INDEX artifacts_analysis_idx ON causal.artifacts (analysis_id);
CREATE INDEX artifacts_type_idx ON causal.artifacts (artifact_type);

CREATE TABLE causal.artifact_parents (
    artifact_id         text NOT NULL REFERENCES causal.artifacts (artifact_id),
    parent_index        integer NOT NULL CHECK (parent_index >= 0),
    parent_artifact_id  text NOT NULL,
    parent_content_hash text NOT NULL CHECK (parent_content_hash ~ '^[a-f0-9]{64}$'),
    PRIMARY KEY (artifact_id, parent_index)
);
