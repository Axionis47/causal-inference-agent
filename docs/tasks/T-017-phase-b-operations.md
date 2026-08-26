# T-017 — Phase B: registered operations, sequential executor, diagnostics

Status: frozen for implementation (isolated-worktree agent; merge by architect)
Owning PRD: PRD-003 §10, §11, §15, §17.4/§17.6 semantics, as amended by §24
Depends on: T-015 (T-016 NOT required — operates on any frame + plan; integration lands in T-019)

## 1. Deliverables

### 1.1 `src/causal/preparation/operations.py` (≤ 250 logical)
Seven registered operation families as pure functions
`(frame, item: PlanItemV1) -> OperationResult(frame, change_counts, imputed_mask_delta,
fitted_params | None)`:
1. `missing_sentinel_normalization` — approved sentinel values → null, exact target
   column + mapping from item parameters.
2. `type_conversion` — strict casts only; a lossy cast raises the typed operation error.
3. `category_normalization` — explicit one-to-one mapping; unmapped level → typed error
   (no silent rare-category combination).
4. `registered_derivation` — closed set: `outcome_observed`, `post_period`,
   `cutoff_side`, nonnull indicator, date component (year/month/day); no expressions.
5. `numeric_median_imputation` — median + missingness indicator column; fit scopes
   `none`/`frozen_frame_blinded`/`pre_treatment_only` restrict the FIT rows; fitted
   median recorded in `fitted_params` (object-store bound by the executor, never traced).
6. `categorical_missing_encoding` — reserved explicit missing level.
7. `estimator_scoped_recipe` — records the recipe; NO fit, frame unchanged.
Every operation writes new columns / new frames only; §10.3 forbidden behaviors are
structurally impossible (no row addition/deletion API exists here).

### 1.2 `src/causal/preparation/executor.py` (≤ 220 logical)
- Sequential topological execution of a committed `PreparationPlanV1`'s items
  (shared `has_cycle` guard; §7 mutation row: fan-out none).
- Per item: expected-input-hash guard (mismatch → typed error, no retry) → operation
  dispatch → new immutable intermediate frame via `shared/frames` → `ExecutionReceiptV1`
  (shared) → postcondition diagnostic run → `tri_agreement` gate → row-invariance assert
  (`row_set_hash` unchanged).
- Idempotent replay: an item whose recorded output already exists with the expected hash
  is skipped with its existing receipt.
- `ExecutionReceiptBundleV1` assembly: receipts, per-column change/missingness counts,
  imputed-cell mask object (row ids × columns as canonical JSON), fitted-params objects.

### 1.3 `src/causal/preparation/diagnostics.py` (≤ 170 logical)
Registered diagnostics from `preparation-diagnostics.v1.json`, each a pure function over
(frame(s), manifest/pack facts) returning `PreparationDiagnosticV1`: row-disposition
reconciliation, key uniqueness/grain, schema/type validation, missingness before/after,
row-set invariance, changed-cell counts by operation and column, contract completeness.
Plus `preview(plan) -> PlanPreviewV1` (expected shape/missingness deltas; no writes).
Unknown diagnostic id fails closed.

### 1.4 Declarative
- `registries/repair-operations.v1.json` (~130): the seven families with parameter
  schemas, phase, targets, pre/postcondition diagnostic ids, forbidden-role guards.
- `registries/preparation-diagnostics.v1.json` (~60): the §15 common set + method
  diagnostics referenced by the T-015 overlay.

### 1.5 Tests (lean — see §2 cap; ~400 logical)
Parametrized per-operation goldens (each family: apply + change counts + mask);
leakage guards (imputation fit scopes never read treatment/outcome/post-treatment
columns — assert via injected role map); executor: order, input-hash mismatch, replay
idempotency, tri_agreement failure stops the chain, row-invariance breach detected;
diagnostics denominators; preview no-write.

## 2. Constraints
Budgets: preparation +≤640 this task; tests +≤400 (D-073 test-headroom cap); declarative +≤200; modules +3.
polars allowed; no langgraph/model/CLI imports; no `causal.design` imports. Fitted
values never appear in logs, events, or exceptions. Pre-code projection + one rethink.

## 3. T-015 absorption notes
- The substrate models live exactly as committed at `762bb5f` (contracts 295, plans 203,
  packs 95, shared/receipts 97) — build against the code, and where this spec's field
  expectations differ from the committed models, the committed models win.
- There is NO migrations ledger table; nothing in this task touches migrations.
- Module count 56/68; largest allowed module 350; largest function 75.
