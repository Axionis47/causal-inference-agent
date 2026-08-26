-- Preparation stage schema (PRD-003 §19, §24; T-015). Operational rows only: task
-- envelopes, previews, and attempt state live here or in traces, never as registered
-- artifact types (Amendment 1 §24.2). No views; they ship with the tasks that need them.
CREATE SCHEMA IF NOT EXISTS preparation;

CREATE TABLE IF NOT EXISTS preparation.preparation_runs (
    stage_run_id          text PRIMARY KEY,
    analysis_id           text NOT NULL,
    graph_thread_id       text NOT NULL,
    state                 text NOT NULL CHECK (state IN
        ('created', 'tracing_preflight', 'running', 'stabilizing', 'frozen', 'preparing',
         'committing', 'completed', 'failed_observability', 'failed')),
    -- The PRD-002 run this preparation run inherits artifacts from; §7.1 forbids
    -- continuing its graph thread, so the two ids are recorded separately.
    design_stage_run_id   text NOT NULL,
    preparation_revision  integer NOT NULL DEFAULT 1 CHECK (preparation_revision >= 1),
    outcome_artifact_id   text REFERENCES causal.artifacts (artifact_id),
    row_set_hash          text CHECK (row_set_hash ~ '^[a-f0-9]{64}$'),
    observability_failure text,
    created_at            timestamptz NOT NULL,
    updated_at            timestamptz NOT NULL,
    UNIQUE (analysis_id, preparation_revision)
);

CREATE INDEX IF NOT EXISTS preparation_runs_analysis_idx
    ON preparation.preparation_runs (analysis_id);

CREATE TABLE IF NOT EXISTS preparation.preparation_artifact_refs (
    stage_run_id   text NOT NULL REFERENCES preparation.preparation_runs (stage_run_id),
    kind           text NOT NULL,
    artifact_id    text NOT NULL REFERENCES causal.artifacts (artifact_id),
    content_hash   text NOT NULL CHECK (content_hash ~ '^[a-f0-9]{64}$'),
    schema_version text NOT NULL,
    PRIMARY KEY (stage_run_id, kind, artifact_id)
);

CREATE TABLE IF NOT EXISTS preparation.preparation_tasks (
    task_id            text PRIMARY KEY,
    stage_run_id       text NOT NULL REFERENCES preparation.preparation_runs (stage_run_id),
    analysis_id        text NOT NULL,
    -- The §7.4 fan-out scope this task covers.
    task_kind          text NOT NULL CHECK (task_kind IN
        ('single_column', 'coupled_columns', 'recipe_group', 'table_wide')),
    phase              text NOT NULL CHECK (phase IN ('stabilization', 'preparation')),
    scope_ids          jsonb NOT NULL,
    envelope_hash      text NOT NULL CHECK (envelope_hash ~ '^[a-f0-9]{64}$'),
    -- The §7.3 stopping states, plus 'dispatched' before the single-shot call returns.
    status             text NOT NULL CHECK (status IN
        ('dispatched', 'proposed', 'needs_dependency', 'design_conflict', 'failed')),
    -- §17.7: one initial response plus at most two corrections.
    attempt_count      integer NOT NULL DEFAULT 1 CHECK (attempt_count BETWEEN 1 AND 3),
    output_artifact_id text REFERENCES causal.artifacts (artifact_id),
    created_at         timestamptz NOT NULL,
    updated_at         timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS preparation_tasks_run_idx
    ON preparation.preparation_tasks (stage_run_id);

CREATE TABLE IF NOT EXISTS preparation.plan_items (
    plan_artifact_id    text NOT NULL REFERENCES causal.artifacts (artifact_id),
    plan_item_id        text NOT NULL,
    stage_run_id        text NOT NULL REFERENCES preparation.preparation_runs (stage_run_id),
    phase               text NOT NULL CHECK (phase IN
        ('stabilization', 'repair', 'derivation', 'imputation', 'diagnostic')),
    -- A committed item unlocks its mutation tool; only a receipted, postcondition-passing
    -- execution may reach 'executed' (§17.6 tri-agreement).
    state               text NOT NULL CHECK (state IN
        ('committed', 'previewed', 'executed', 'failed', 'superseded')),
    receipt_artifact_id text REFERENCES causal.artifacts (artifact_id),
    created_at          timestamptz NOT NULL,
    updated_at          timestamptz NOT NULL,
    PRIMARY KEY (plan_artifact_id, plan_item_id)
);

CREATE INDEX IF NOT EXISTS plan_items_run_idx ON preparation.plan_items (stage_run_id);
