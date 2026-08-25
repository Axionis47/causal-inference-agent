# T-008 — Intake capture, catalog, coordinator, and handoff

Status: frozen at commit time. Governing sources: PRD-001 §3.1, §4, §5, §9–§13,
§15.1; SYSTEM-CONTRACT §3, §4, §8, §10.1. Depends on T-005, T-006, T-007.

## Goal

Finish PRD-001: a deterministic intake coordinator that takes one validated
`IntakeSubmissionV1` to a committed `IntakeOutcome`, writes the `catalog`
schema, and records the PRD-002 handoff. No model call anywhere.

## Deliverables

1. `src/causal/intake/kaggle.py` — `KaggleClientProtocol` (dataset status,
   metadata, file listing, archive download) and a capture layer that freezes
   one provider snapshot into the `KaggleCapture` payload. No credential field
   exists anywhere in the payload or the Protocol; the live adapter is a later
   task (D-034).
2. `registries/kaggle-field-classes.v1.json` — static field-classification
   registry (PRD-001 §5.7) mapping provider fields to context classes, plus a
   loader in `src/causal/intake/fields.py` that builds `source_field_index`
   rows from a capture (D-030).
3. `src/causal/intake/semantic.py` — deterministic `EvidenceBundle` and
   `SemanticMap` builders (D-032).
4. `migrations/0003_catalog.sql` — `catalog` schema: `runs`, `datasets`,
   `resources`, `source_field_index`, and exactly the five narrow views
   (`semantic_available`, `semantic_missing`, `structural_manifest`,
   `measured_fact_manifest`, `provenance_manifest`). Deliberately no
   `all_context` view (D-029).
5. `src/causal/intake/catalog.py` — `CatalogStore`: run identity and
   idempotency, dataset and resource rows, field-index writes, and the §11.1(7)
   single-transaction finalization of `intake_status` +
   `intake_outcome_artifact_id`.
6. `src/causal/intake/coordinator.py` — `IntakeCoordinator.run(submission)`
   executing the §4 flow to a committed `IntakeOutcome`, plus
   `open_handoff(analysis_id, intake_outcome_artifact_id, receiving_stage_run_id)`
   building the PRD-002 `HandoffManifestV1` mechanically for usable/partial
   outcomes only (Amendment 1, D-037).
7. Tests covering EV-P1-001, EV-P1-002, and EV-P1-005 fixture focuses at the
   unit/integration layer (frozen fixture clients; dockerized Postgres+MinIO).

## Flow (frozen)

record QuestionRecord → capture provider snapshot (KaggleCapture) → download
and hash archive → admit (T-007 ArchiveSafety) → resource rows for every entry
→ SourceManifest → profile admitted tables (TableProfile each) ∥ extract
admitted documents → EvidenceBundle → SemanticMap → source_field_index rows →
IntakeOutcome → finalize runs row (one transaction) → handoff record
(usable/partial only). Refusal paths (§12) still commit QuestionRecord, a
capture when one exists, and a refused IntakeOutcome; they never record a
handoff.

Stage-run states: created → tracing_preflight → running → committing →
completed, with failures mapping to failed. The LangSmith preflight/flush gate
remains deferred (D-020); the coordinator passes through tracing_preflight.

## Identities (D-031)

- `analysis_id` = `an-` + first 16 hex of sha256(idempotency_key).
- `dataset_id` = `kaggle:{owner}/{slug}@{version}`.
- `artifact_id` = `{kind}:{analysis_id}:` + first 16 hex of the payload hash.
- Artifact payloads contain no wall-clock values; envelope `created_at_utc`
  comes from an injected clock. Same inputs replay to the same IDs and hashes.

## Idempotency (§3.1)

`catalog.runs` stores `idempotency_key` (unique) and the submission's content
hash. Same key + same hash: return the recorded analysis (its outcome if
committed; otherwise re-run under a new `stage_run_id`, D-035 — deterministic
artifact IDs make recommits §8.2 replay no-ops). Same key + different hash:
raise a blocker (`blocker.raised`), never create a second analysis.

## Outcome status rule (D-033)

- `refused`: capture failure, unresolvable version, archive refused by safety,
  hash conflict, or zero profiled tables.
- `partial`: technically usable but any resource is unreadable/excluded/failed,
  or no semantic field is evidenced.
- `usable`: otherwise. All discovered resources have terminal `parse_status`.

## Semantic map rules (D-032)

Column `meaning` ← provider column description (evidenced with evidence ID;
`empty` when offered blank; `not_offered` when absent). Column
`missing_sentinel` ← `hypothesis` from the T-007 profiler when present. Every
other column slot and all seven dataset slots are `not_offered` in v0: mapping
free text onto slots is interpretation and belongs to PRD-002 design agents.
Raw semantic texts (title, subtitle, description, keywords, version notes,
file/column descriptions, admitted document text ≤ 1 MiB per document) are
carried in `EvidenceBundle` with stable evidence IDs and indexed in
`source_field_index`, so nothing is lost by the conservative slot mapping.

## Registry corrections (D-027, D-036)

- `IntakeOutcome.required_parent_types` becomes `["QuestionRecord"]` with the
  full chain (`SourceManifest`, `TableProfile`, `EvidenceBundle`,
  `SemanticMap`) moved to `optional_parent_types`: an early refusal has no
  chain to cite. The coordinator itself requires the full chain for
  usable/partial outcomes.
- `resources.parse_status` vocabulary is PRD-001's five values (`parsed`,
  `excluded`, `unreadable`, `unsafe`, `failed`); `withheld` classifications map
  to `excluded` with a reason.

## Storage split (D-028, D-029)

Canonical-JSON artifacts go through `ArtifactCommitter` with envelopes. Raw
bytes (the archive, extracted source files) are content-addressed objects
written via `ObjectStore.put_if_absent` without envelopes; `catalog.resources`
rows point at them. The PRD's `artifacts` table is realized by the shared
`causal.artifacts`; run operational state stays in `causal.stage_runs`.

## Events (§13)

`stage.started/completed/failed` for the run; `tool.*` around provider
operations (eval IDs: EV-P1-002); `task.*` per resource processed (EV-P1-003);
`artifact.committed` via the committer (EV-P1-005 at the outcome commit);
`handoff.accepted/rejected` via the T-006 gate; `blocker.raised` on idempotency
conflict. No event carries credentials, raw provider bodies, or table rows.

## Acceptance for this task

1. End-to-end fixture run reaches `usable`, and repeating it returns identical
   artifact IDs and hashes (PRD acceptance 1, 26).
2. Conflicting idempotency key raises a blocker without a second analysis.
3. Unsafe archive → `refused` outcome, no handoff record, artifacts preserved.
4. Zero profiled tables → `refused`; weak semantics → `partial` with handoff.
5. Every archive entry has a `catalog.resources` row with terminal status.
6. The five views answer availability queries; no `all_context` view exists.
7. `HandoffGate.accept` (receiving component `design-harness`, allowed
   outcomes usable/partial) accepts the recorded manifest from `analysis_id` +
   `intake_outcome_artifact_id` alone.
8. Credential canary: a fixture client carrying a token attribute produces no
   artifact payload, event line, or catalog row containing it.
9. ruff, mypy --strict, and the §14.1 budget gate pass; intake stays within
   its 1,200-line allocation.

## Amendment 1 (pre-implementation; D-037)

The producer does not persist the handoff manifest: T-006's `HandoffGate.accept`
records it at receipt, and §11.1 opens the handoff from `analysis_id` +
`intake_outcome_artifact_id` alone. The coordinator instead exposes
`open_handoff(...)`, which rebuilds the manifest deterministically
(`ho:{analysis_id}:{outcome-hash16}`) from the catalog and refuses
(`handoff_unavailable`) for refused, missing, or mismatched outcomes. The
measured-fact surface gains index rows: one `measured`-class row per profiled
column, whose pointer names the `TableProfile` artifact and the column path.

## Amendment 2 (pre-implementation; D-038)

The shared docker fixtures move from `tests/shared/conftest.py` to
`tests/conftest.py` so intake integration tests reuse them unchanged.

## Amendment 3 — structural rows for profiled columns (2026-08-25, pilot finding, D-064)

A zero-column-metadata Kaggle dataset (0/87 semantic slots offered) produced an empty
`catalog.structural_manifest` for its table: per-column rows existed only in the
`measured` class, so the design stage's `structural_inventory` — and therefore every
model task's column list — was empty. A profiled column's existence is structural fact
independent of provider metadata.

Fix (intake scope, ≤ +3 logical lines): `semantic.measured_index_rows` additionally
emits, per profiled column, one `FieldIndexRow("column", table, column, "presence",
ContextClass.STRUCTURAL, SemanticStatus.EVIDENCED, 0, pointer)` alongside the measured
row (same profile pointer). `entry._structural_inventory` then lists every profiled
column with `dtype="undeclared"` unless a provider type slot is also evidenced —
already its semantics; no design change.

Tests (≤ 25 lines): a profiles fixture with no provider metadata asserts one structural
`presence` row per column and that the design manifest's structural inventory over
those rows lists all columns.
