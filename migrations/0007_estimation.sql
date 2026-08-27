-- Estimation stage schema (PRD-004 §21; T-024). Operational rows, artifact pointers,
-- parent and contribution-mask indexes only: every array, fit, prediction, weight, and
-- figure payload stays a restricted object (§21). No views; they ship with their task.
CREATE SCHEMA IF NOT EXISTS estimation;

CREATE TABLE IF NOT EXISTS estimation.estimation_runs (
    stage_run_id             text PRIMARY KEY,
    analysis_id              text NOT NULL,
    -- §26.1 keeps the thread id as a trace identity; PRD-004 never continues PRD-003's.
    graph_thread_id          text NOT NULL,
    state                    text NOT NULL CHECK (state IN
        ('created', 'tracing_preflight', 'running', 'estimating', 'judging',
         'committing', 'completed', 'failed_observability', 'failed')),
    -- The PRD-003 run whose prepared bundle this run inherits.
    preparation_stage_run_id text NOT NULL,
    estimation_revision      integer NOT NULL DEFAULT 1 CHECK (estimation_revision >= 1),
    context_manifest_id      text REFERENCES causal.artifacts (artifact_id),
    plan_artifact_id         text REFERENCES causal.artifacts (artifact_id),
    outcome_artifact_id      text REFERENCES causal.artifacts (artifact_id),
    row_set_hash             text CHECK (row_set_hash ~ '^[a-f0-9]{64}$'),
    -- §16.1 overall ceiling, §5.2 conflict code, and the stable failure code.
    overall_ceiling          text,
    conflict_code            text,
    error_code               text,
    observability_failure    text,
    created_at               timestamptz NOT NULL,
    updated_at               timestamptz NOT NULL,
    UNIQUE (analysis_id, estimation_revision)
);

CREATE INDEX IF NOT EXISTS estimation_runs_analysis_idx
    ON estimation.estimation_runs (analysis_id);

CREATE INDEX IF NOT EXISTS estimation_runs_conflict_idx
    ON estimation.estimation_runs (conflict_code) WHERE conflict_code IS NOT NULL;

CREATE INDEX IF NOT EXISTS estimation_runs_failure_idx
    ON estimation.estimation_runs (error_code) WHERE error_code IS NOT NULL;

CREATE TABLE IF NOT EXISTS estimation.estimation_artifact_refs (
    stage_run_id   text NOT NULL REFERENCES estimation.estimation_runs (stage_run_id),
    kind           text NOT NULL,
    artifact_id    text NOT NULL REFERENCES causal.artifacts (artifact_id),
    content_hash   text NOT NULL CHECK (content_hash ~ '^[a-f0-9]{64}$'),
    schema_version text NOT NULL,
    PRIMARY KEY (stage_run_id, kind, artifact_id)
);

CREATE INDEX IF NOT EXISTS estimation_artifact_refs_artifact_idx
    ON estimation.estimation_artifact_refs (artifact_id);

CREATE TABLE IF NOT EXISTS estimation.contribution_mask_index (
    mask_artifact_id      text PRIMARY KEY REFERENCES causal.artifacts (artifact_id),
    stage_run_id          text NOT NULL REFERENCES estimation.estimation_runs (stage_run_id),
    -- §6.2: which calculation the mask serves, the registered rule, and the frozen row set.
    calculation_id        text NOT NULL,
    mask_rule_id          text NOT NULL,
    parent_row_set_hash   text NOT NULL CHECK (parent_row_set_hash ~ '^[a-f0-9]{64}$'),
    included_rows         integer NOT NULL CHECK (included_rows >= 0),
    noncontributing_rows  integer NOT NULL CHECK (noncontributing_rows >= 0),
    created_at            timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS contribution_mask_run_idx
    ON estimation.contribution_mask_index (stage_run_id);
