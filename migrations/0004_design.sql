-- Design stage schema (PRD-002 §21; T-009). No views; design views ship with the tasks
-- that need them.
CREATE SCHEMA design;

CREATE TABLE design.design_runs (
    stage_run_id          text PRIMARY KEY,
    analysis_id           text NOT NULL,
    graph_thread_id       text NOT NULL,
    design_revision       integer NOT NULL CHECK (design_revision >= 1),
    state                 text NOT NULL CHECK (state IN
        ('created', 'tracing_preflight', 'running', 'waiting_for_user',
         'committing', 'completed', 'failed_observability', 'failed')),
    selected_table        text,
    method_id             text,
    outcome_artifact_id   text REFERENCES causal.artifacts (artifact_id),
    observability_failure text,
    created_at            timestamptz NOT NULL,
    updated_at            timestamptz NOT NULL,
    UNIQUE (analysis_id, design_revision)
);

CREATE TABLE design.design_artifact_refs (
    artifact_id     text PRIMARY KEY REFERENCES causal.artifacts (artifact_id),
    analysis_id     text NOT NULL,
    design_revision integer NOT NULL CHECK (design_revision >= 1),
    kind            text NOT NULL,
    content_hash    text NOT NULL CHECK (content_hash ~ '^[a-f0-9]{64}$'),
    schema_version  text NOT NULL,
    approval_bound  boolean NOT NULL DEFAULT false
);

CREATE TABLE design.design_context_manifests (
    artifact_id                text PRIMARY KEY REFERENCES causal.artifacts (artifact_id),
    analysis_id                text NOT NULL,
    design_revision            integer NOT NULL CHECK (design_revision >= 1),
    content_hash               text NOT NULL CHECK (content_hash ~ '^[a-f0-9]{64}$'),
    intake_outcome_artifact_id text NOT NULL REFERENCES causal.artifacts (artifact_id),
    registry_versions          jsonb NOT NULL
);

CREATE TABLE design.causal_graph_views (
    artifact_id       text PRIMARY KEY REFERENCES causal.artifacts (artifact_id),
    analysis_id       text NOT NULL,
    design_revision   integer NOT NULL CHECK (design_revision >= 1),
    renderer_version  text NOT NULL,
    validation_status text NOT NULL
);

CREATE TABLE design.design_tasks (
    task_id               text PRIMARY KEY,
    analysis_id           text NOT NULL,
    stage_run_id          text NOT NULL,
    design_revision       integer NOT NULL CHECK (design_revision >= 1),
    task_kind             text NOT NULL,
    scope                 jsonb NOT NULL,
    envelope_hash         text NOT NULL CHECK (envelope_hash ~ '^[a-f0-9]{64}$'),
    prompt_version        text NOT NULL,
    model_profile_version text NOT NULL,
    status                text NOT NULL CHECK (status IN
        ('dispatched', 'complete', 'needs_context', 'conflict', 'refused', 'failed')),
    output_artifact_id    text REFERENCES causal.artifacts (artifact_id),
    attempts              integer NOT NULL DEFAULT 1 CHECK (attempts >= 1),
    UNIQUE (analysis_id, design_revision, task_id)
);

CREATE TABLE design.context_requirements (
    analysis_id                  text NOT NULL,
    design_revision              integer NOT NULL CHECK (design_revision >= 1),
    requirement_id               text NOT NULL,
    scope_kind                   text NOT NULL,
    scope_id                     text NOT NULL,
    criticality                  text NOT NULL CHECK (criticality IN
        ('blocking', 'supporting')),
    missing_action               text NOT NULL CHECK (missing_action IN
        ('ask_user', 'retain_as_sensitivity', 'refuse')),
    state                        text NOT NULL CHECK (state IN
        ('open', 'resolved', 'unknown_accepted', 'refused')),
    attempted_evidence           jsonb NOT NULL,
    resolving_answer_artifact_id text REFERENCES causal.artifacts (artifact_id),
    PRIMARY KEY (analysis_id, design_revision, requirement_id)
);

CREATE TABLE design.delivery_capacity_checks (
    artifact_id                   text PRIMARY KEY REFERENCES causal.artifacts (artifact_id),
    analysis_id                   text NOT NULL,
    design_revision               integer NOT NULL CHECK (design_revision >= 1),
    status                        text NOT NULL CHECK (status IN ('pass', 'fail')),
    cardinalities                 jsonb NOT NULL,
    capacity_registry_version     text NOT NULL,
    visualization_catalog_version text NOT NULL
);

CREATE TABLE design.design_approvals (
    artifact_id     text PRIMARY KEY REFERENCES causal.artifacts (artifact_id),
    analysis_id     text NOT NULL,
    design_revision integer NOT NULL CHECK (design_revision >= 1),
    decision        text NOT NULL CHECK (decision IN
        ('approved', 'changes_requested', 'declined')),
    approved_hashes jsonb NOT NULL,
    decided_at      timestamptz NOT NULL
);
