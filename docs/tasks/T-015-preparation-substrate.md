# T-015 — Preparation substrate: contracts, plans, pack overlay, receipts, registry rows, migration

Status: frozen for implementation
Owning PRD: PRD-003 as amended by Amendment 1 (V1-lite, §24) — read §24 FIRST, it governs
Depends on: PRD-002 wave (T-009..T-014, all ACCEPTED)

## 1. Deliverables

### 1.1 `src/causal/preparation/contracts.py` (≤ 260 logical lines)

Frozen, `extra="forbid"`, `strict=True` pydantic models (mirror `design/contracts.py` style):

- `RowDisposition` StrEnum — exactly the nine §6.2 values (`retained`,
  `retained_with_missingness`, `not_eligible_population`, `not_eligible_timeframe`,
  `unusable_corrupt_record`, `unusable_required_identity`, `unusable_required_role`,
  `unusable_grain_violation`, `unresolved_conflict`).
- `PreparationContextManifestV1` — §7.2 fields, lite: selected CSV ref (artifact id, hash,
  object locator, parser profile id), question/population/timeframe/treatment/outcome/
  comparator/estimand ids, selected method + pack_version, measurement-map and role-ledger
  refs, column→concept and column→role maps, protected/permitted-repair/
  permitted-imputation column sets, approved grain + key + schema id, eligibility and
  unusable-row rule ids, structural requirements, permitted operation/diagnostic registry
  ids, deletion-impact dimensions, registry_versions map, recipient map, manifest hash.
- `DispositionCountV1` (disposition → count) and `DimensionImpactV1` (dimension id,
  retained/excluded counts per level, warning codes).
- `StabilizationRecordV1` — §24.2 consolidation: source-row-index summary (row count,
  parse-warning counts, index object locator + hash), eligibility evaluation summary
  (rule id → counts), disposition ledger summary (DispositionCountV1 list + ledger object
  locator + hash), impact (DimensionImpactV1 list), method-structure validation result,
  freeze block (ordered retained-set object locator + hash, retained row count, unique
  units, `row_set_hash`), pre/post-stabilization diagnostic results, versions.
- `PreparationDiagnosticV1` — §15 fields: diagnostic id + version, frame stage
  (`source`|`stabilized`|`prepared`), input refs + hashes, columns read, totals/used/
  unused with reason counts, values, warnings, terminal status, implementation version.
- `StabilizedFrameV1` / `PreparedFrameV1` — frame metadata artifacts: parent refs,
  column schema (name, dtype, prepared-from), row count, `row_set_hash`, content-addressed
  frame object locator + hash, writer version.
- `PreparedFrameBundleV1` — refs+hashes for: selected table, experiment design, runnable
  frame contract, capacity check, stabilization record, stabilized frame, prepared frame,
  execution receipt bundle; one `row_set_hash`; version map. (Amendment 1 satisfies §5.1
  through these consolidated parents.)
- `PreparationOutcomeV1` — §5.2 statuses (`prepared`, `design_conflict`, `not_runnable`,
  `failed_observability`, `failed`); refs to manifest, bundle (required iff `prepared`),
  conflict (required iff `design_conflict`); stage_run_id, graph_thread_id, error_code.
- `DesignConflictV1` (+ `DesignConflictDraftV1` = same fields minus ids/hashes stamped by
  the harness) — §16 fields: stable conflict code, failed rule id, affected counts,
  evidence refs, why no permitted operation resolves it, material design fields needing
  revision, recommended PRD-002 action (`ask_user`|`revise_design`|`refuse`).

### 1.2 `src/causal/preparation/plans.py` (≤ 170 logical lines)

- `FitScope` StrEnum — `none`, `frozen_frame_blinded`, `pre_treatment_only`,
  `cross_fit_training_fold` (§11.2).
- `PlanItemV1` — plan_item_id, phase (`stabilization`|`repair`|`derivation`|`imputation`|
  `diagnostic`), operation id + version, target columns, output column (optional),
  parameters (closed dict), fit scope, predicted missingness change, dependencies
  (plan_item_ids), postcondition ids, rationale evidence refs.
- `PreparationPlanV1` — §24.2: plan_revision, phase (`stabilization`|`preparation`),
  ordered `PlanItemV1` tuple, eligibility/unusable rule ids (stabilization phase),
  estimator-scoped recipe entries (`EstimatorScopedRecipeV1`: recipe id, targets,
  fit scope `cross_fit_training_fold`, method pack binding — recorded, never fitted),
  task-grouping metadata (§7.4 group kind per item group), parent refs, versions.
- `PreparationTaskContextV1` — §7.3 payload fields (task id, phase, scope kind/ids,
  gap codes, roles/concepts/timing/protected refs, permitted operation ids, dependency
  ids, budgets, stopping states) and `PreparationTaskDraftV1` (proposed `PlanItemV1`
  tuple + optional `DesignConflictDraftV1` + notes).
- `ExecutionReceiptBundleV1` — ordered receipt tuple (shared `ExecutionReceiptV1`),
  per-column change counts, missingness before/after per column, imputed-cell mask
  object locator + hash, `row_set_hash` asserted, parent refs.
- Preview model `PlanPreviewV1` — expected shape/missingness deltas per item, no writes.

### 1.3 `src/causal/preparation/packs.py` (≤ 100 logical lines)

Loader for `registries/method-pack-preparation.v1.json` (sidecar overlay; design's frozen
loader untouched). Row model `PreparationPackV1` keyed by (method_id, pack_version):
permitted disposition-rule ids, protected column roles, required observed-role rules,
row and unique-unit minimums, structure/cell-support gates, permitted repair-operation
ids, permitted imputation targets + fit scopes, required missingness indicators,
required post-stabilization and post-repair diagnostic ids, prepared-frame schema id,
estimator input contract id (§13). The loader fails closed (`RegistryError`,
`invalid_registry_file` / `unknown_method_pack`) when a row's (method_id, pack_version)
is absent from `registries/method-packs.v1.json` — validate against the RAW JSON file
(no import from `causal.design`), or a row is missing/unknown. All four methods present.

### 1.4 `src/causal/shared/receipts.py` (≤ 100 logical lines, shared scope)

- `ExecutionReceiptV1` — §17.6 lite fields: stage_run_id, plan/plan_item/operation ids +
  versions, exact input ref + hash, exact output ref + hash, parameters hash,
  before/after shape, `row_set_hash` before/after (must match), examined/changed/derived/
  imputed counts, warning + error codes, attempt id, idempotency key, terminal status,
  started/finished timestamps (injected clock).
- `tri_agreement(receipt, output_envelope, postcondition) -> tuple[str, ...]` — the
  §17.6 gate: returns the stable error codes (empty = pass) checking receipt terminal
  status, output artifact id + hash equality, postcondition terminal pass, and row-hash
  invariance. Postcondition input is a `PreparationDiagnosticV1`-shaped mapping; keep the
  checker generic over a small Protocol so PRD-004 reuses it.

### 1.5 Declarative

- `registries/artifact-types.v1.json` — append EXACTLY these nine rows (producer
  `preparation-harness` unless stated; schema versions `<kebab-name>.v1`):
  1. `PreparationContextManifest` — readers [preparation-harness]; parents required
     [TableSelection, ExperimentDesign, RunnableFrameContract, DeliveryCapacityCheck];
     statuses [committed]; destinations [preparation].
  2. `StabilizationRecord` — readers [preparation-harness, estimation-harness]; parents
     [PreparationContextManifest]; statuses [committed]; destinations [preparation, estimation].
  3. `StabilizedFrame` — readers [preparation-harness, estimation-harness]; parents
     [StabilizationRecord]; statuses [committed]; destinations [preparation, estimation].
  4. `PreparationPlan` — readers [preparation-harness]; parents
     [PreparationContextManifest]; optional [StabilizationRecord]; statuses [committed];
     destinations [preparation].
  5. `ExecutionReceiptBundle` — readers [preparation-harness, estimation-harness];
     parents [PreparationPlan]; statuses [committed]; destinations [preparation, estimation].
  6. `PreparedFrame` — readers [preparation-harness, estimation-harness]; parents
     [StabilizedFrame, ExecutionReceiptBundle]; statuses [committed]; destinations
     [preparation, estimation].
  7. `PreparedFrameBundle` — readers [preparation-harness, estimation-harness]; parents
     [TableSelection, ExperimentDesign, RunnableFrameContract, DeliveryCapacityCheck,
     StabilizationRecord, StabilizedFrame, PreparedFrame, ExecutionReceiptBundle];
     statuses [prepared]; destinations [estimation]. (Matches the amended SC §3.1 row.)
  8. `PreparationOutcome` — readers [preparation-harness, estimation-harness,
     design-harness]; parents [PreparationContextManifest]; optional [PreparedFrameBundle,
     DesignConflict]; statuses [prepared, design_conflict, not_runnable,
     failed_observability, failed]; destinations [preparation, estimation, design].
  9. `DesignConflict` — readers [preparation-harness, design-harness]; parents
     [PreparationContextManifest]; statuses [open, resolved, refused]; destinations
     [preparation, design].
- `migrations/0006_preparation.sql` (~130 lines) — `preparation` schema:
  `preparation_runs` (stage_run_id PK, analysis_id, graph_thread_id, state,
  design_stage_run_id, outcome_artifact_id NULL, row_set_hash NULL, created/updated),
  `preparation_artifact_refs` (stage_run_id, kind, artifact_id, content_hash; PK
  (stage_run_id, kind, artifact_id)), `preparation_tasks` (task_id PK, stage_run_id,
  task_kind, scope_ids, envelope_hash, status, attempt_count, output_artifact_id NULL),
  `plan_items` (plan_item_id, plan_artifact_id, stage_run_id, state, receipt_artifact_id
  NULL, PK (plan_artifact_id, plan_item_id)). Idempotent per the 000x pattern; registered
  in `public.causal_migrations`.
- `registries/method-pack-preparation.v1.json` (~220 lines) — one row per existing pack
  (randomized_experiment, aipw, did, sharp_rdd) with §12/§13 values derived from the
  PRD's method tables (e.g. RCT: protected roles treatment/assignment/outcome-attrition,
  never-imputed compliance; AIPW: one-row-per-unit minimum, confounder imputation
  `cross_fit_training_fold`; DiD: group-time support gates, pre-treatment-only
  imputation; RDD: running-variable/cutoff protected, blinded pooled strategies only).

### 1.6 Tests (~350 logical lines, `tests/preparation/`)

- `test_contracts.py` — round-trips + canonical hashing per model; disposition enum
  closed; PreparedFrameBundle rejects mismatched `row_set_hash`; outcome status/ref
  coupling (prepared requires bundle; design_conflict requires conflict).
- `test_plans.py` — plan item dependency shapes; `has_cycle` over item deps; fit-scope
  enum; recipe-never-fitted invariant (no fitted-values field exists).
- `test_packs.py` — overlay loads all four; unknown pack_version fails closed; binding
  against a tampered method-packs file fails closed.
- `test_receipts.py` (tests/shared/) — tri_agreement pass + each failure code.
- `test_registry_rows.py` — the nine rows load through `ArtifactTypeRegistry`; reader/
  destination/parent expectations exact (mirror tests/design/test_registry_rows.py).
- `test_migration.py` — 0006 applies idempotently against docker Postgres; tables exist.

## 2. Constraints

- Budgets: preparation ≤ 630 total this task; shared +≤100 (→ ≤ 1,977/2,500); modules
  +5 (→ 56/68); tests +≤360 (→ ≤ 6,900/9,000); declarative +≤560 (→ ≤ 2,725/3,500).
  Every module ≤ 350 logical lines, every function ≤ 75. Pre-code projection required
  (SC §14.1.2); one rethink available.
- No langgraph, model, polars, or CLI imports anywhere in this task. `contracts.py` and
  `plans.py` import only pydantic + shared. No cross-import from `causal.design`
  (registry JSON files are read as data).
- Registry file stays version `artifact-types.v1` (append-only, D-012).
- Style: mirror the design substrate modules (one-line module docstrings citing sections
  and decisions; frozen models; `Identity` types from shared contracts).
