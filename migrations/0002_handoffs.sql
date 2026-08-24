-- Handoff visibility (SYSTEM-CONTRACT §3, §8.1; task T-006).
CREATE TABLE causal.handoffs (
    handoff_id                 text PRIMARY KEY,
    schema_version             text NOT NULL,
    analysis_id                text NOT NULL,
    producing_stage_run_id     text NOT NULL,
    receiving_stage_run_id     text NOT NULL,
    originating_outcome        text NOT NULL,
    approval_ids               text[] NOT NULL,
    registry_version           text NOT NULL,
    compatibility_version      text NOT NULL,
    receiver_validation_result text,
    receiver_error_codes       text[] NOT NULL DEFAULT '{}',
    created_at_utc             timestamptz NOT NULL,
    accepted_at_utc            timestamptz
);

CREATE INDEX handoffs_analysis_idx ON causal.handoffs (analysis_id);

CREATE TABLE causal.handoff_entries (
    handoff_id   text NOT NULL REFERENCES causal.handoffs (handoff_id),
    entry_index  integer NOT NULL CHECK (entry_index >= 0),
    artifact_id  text NOT NULL,
    content_hash text NOT NULL CHECK (content_hash ~ '^[a-f0-9]{64}$'),
    PRIMARY KEY (handoff_id, entry_index)
);
