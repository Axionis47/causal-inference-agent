# T-024 — Estimation coordinator substrate: harness, engine, schema migration

Status: frozen for implementation
Owning PRD: PRD-004 §6.2, §6.4, §7, §14, §15, §16.1, §19 (per §26), §21
Depends on: T-022, T-023

## 1. Deliverables

1. `src/causal/estimation/harness.py` (≤ 270 logical) — mirrors
   `preparation/harness.py`: `EstimationState` TypedDict (§19 allowlist fields
   only), `EstimationDeps` (catalog/object readers, committer, emitter, tracer,
   pack registry, gateway ONLY for the claim task), `EstimationRunResult`,
   `HarnessBase` with: flush-gated commits, wall runner, event emission bound to
   EV-P4 eval ids, conflict draft-and-commit routing (DesignConflict → PRD-002,
   never user), frame reads through shared readers, `estimation.estimation_runs`
   SQL (UNIQUE (analysis_id, estimation_revision); terminal on every exit path —
   the `preparation_runs` idiom).
2. `src/causal/estimation/engine.py` (≤ 210):
   - contribution-mask builder: registered mask rules (outcome_observed,
     bandwidth, event_cell, all_rows) → bit vector as restricted object +
     `AnalysisContributionMaskV1` with exact counts; parent `row_set_hash` bound.
   - `EstimatorAdapter` Protocol: `fit(frame_view, plan, pack, overrides=None) ->
     AdapterResult` (primary items + harvest values + fit handle for sensitivity
     reuse); adapters never see undeclared columns (view built from plan roles).
   - sequential evidence runner (§26.1): registered order, every diagnostic /
     sensitivity reaches a visible terminal result; failures recorded, never
     omitted; severity applier maps values → {acceptable, warning, invalidating,
     descriptive} from pack threshold rows (generic, one function).
   - sensitivity loop: branch row → param-delta overrides → adapter re-invoke →
     comparison rule (direction/magnitude/interval-overlap) → SensitivityResultV1.
   - shared balance helper (grouped means/std diffs; reused by RCT and AIPW).
   - `JudgmentCeilingV1` calculator (§16.1 table; rule ids + evidence refs).
   - `NumericalEnvironmentManifestV1` builder (importlib.metadata versions,
     platform, seeds, threads, tolerances).
   - generic figure-data committer wrapping adapter payload functions into
     `FigureDataArtifactV1` (+ EvidenceBundle assembly for all three kinds).
3. `migrations/0007_estimation.sql` (~60 declarative) — `estimation` schema:
   estimation_runs, artifact-pointer/parent indexes per §21; follows 0006's
   apply_migrations idempotency pattern (IF NOT EXISTS + re-apply test).

## 2. Tests (≤ 350 logical)

Dockerized (shared conftest fixtures): run-row lifecycle incl. crash re-entry at
revision n+1; commit/replay idempotence (recommit = no-op, same ids); migration
re-apply. Pure: mask builder counts/hash stability + row_set_hash mismatch refusal;
severity applier over every §14.2 severity; ceiling calculator truth table (each
§16.1 condition → its cap; overall = most restrictive); sensitivity comparison
rules; env-manifest determinism (two builds agree modulo none).

## 3. Constraints

estimation ≤ 1,440 total after (+480); declarative ≤ +60; tests ≤ 350; modules +2.
No LangGraph imports anywhere in scope. Wave rethink shared. Committed code wins;
record deviations.
