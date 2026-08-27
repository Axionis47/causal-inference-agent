-- Presentation stage schema (PRD-005 §18; T-032). One small run row per presentation revision:
-- the §18 record's five upstream ids, handoff ids, catalog/theme/font/compiler/renderer
-- identities, current artifact ids, counters, stable failure ids, and the terminal outcome all
-- travel inside `run_record` as the canonical PresentationRunV1 payload, and every manifest,
-- plan, specification, render, table, summary, report, and bundle stays an immutable artifact in
-- `causal.artifacts` with its bytes in the shared object store. `state` is PRD-005 §1's closed
-- status vocabulary, enforced by PresentationOutcomeStatus rather than duplicated in a CHECK.
-- The UNIQUE key below is also the index the newest-revision lookup reads.
CREATE SCHEMA IF NOT EXISTS presentation;
CREATE TABLE IF NOT EXISTS presentation.runs (
    stage_run_id text PRIMARY KEY, analysis_id text NOT NULL, state text NOT NULL,
    presentation_revision integer NOT NULL CHECK (presentation_revision >= 1),
    bundle_artifact_id text REFERENCES causal.artifacts (artifact_id), error_code text,
    run_record jsonb NOT NULL, created_at timestamptz NOT NULL, updated_at timestamptz NOT NULL,
    UNIQUE (analysis_id, presentation_revision));
