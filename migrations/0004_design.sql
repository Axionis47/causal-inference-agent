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
    created_at            timestamptz NOT NULL,
    updated_at            timestamptz NOT NULL,
    UNIQUE (analysis_id, design_revision)
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
    resolving_fact_id            text,
    PRIMARY KEY (analysis_id, design_revision, requirement_id, scope_id)
);

CREATE TABLE design.accepted_facts (
    accepted_fact_id       text PRIMARY KEY,
    analysis_id            text NOT NULL,
    design_revision        integer NOT NULL CHECK (design_revision >= 1),
    requirement_id         text NOT NULL,
    scope_id               text NOT NULL,
    value                  jsonb NOT NULL,
    value_schema           text NOT NULL,
    source_kind            text NOT NULL CHECK (source_kind IN
        ('user', 'document', 'measurement', 'interpretation', 'derived')),
    evidence_ids           jsonb NOT NULL,
    evidence_class         text NOT NULL,
    support_relation       text NOT NULL CHECK (support_relation IN
        ('direct', 'corroborating')),
    acceptance_status      text NOT NULL CHECK (acceptance_status IN
        ('accepted', 'superseded', 'conflicting')),
    is_current             boolean NOT NULL,
    origin_revision        integer NOT NULL CHECK (origin_revision >= 1),
    origin_reference_id    text NOT NULL,
    origin_reference_hash  text NOT NULL CHECK (origin_reference_hash ~ '^[a-f0-9]{64}$'),
    inherited_from_fact_id text REFERENCES design.accepted_facts (accepted_fact_id),
    supersedes_fact_id     text REFERENCES design.accepted_facts (accepted_fact_id),
    created_at             timestamptz NOT NULL,
    UNIQUE (analysis_id, design_revision, accepted_fact_id)
);

CREATE UNIQUE INDEX accepted_facts_one_current
    ON design.accepted_facts (analysis_id, design_revision, requirement_id, scope_id)
    WHERE is_current;

ALTER TABLE design.context_requirements
    ADD CONSTRAINT context_requirement_resolving_fact_fk
    FOREIGN KEY (resolving_fact_id) REFERENCES design.accepted_facts (accepted_fact_id);

ALTER TABLE design.context_requirements
    ADD CONSTRAINT resolved_requirement_has_fact
    CHECK ((state = 'resolved') = (resolving_fact_id IS NOT NULL));
