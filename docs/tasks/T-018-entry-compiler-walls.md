# T-018 — Entry gate, gap triage + plan compiler, six walls

Status: frozen for implementation
Owning PRD: PRD-003 §4, §7.2, §7.4, §7.5, §14 (as §24.3), §16; SC §3 receiver rule
Depends on: T-015, T-016, T-017

## 1. Deliverables

### 1.1 `src/causal/preparation/entry.py` (≤ 260 logical)
- Replay-safe handoff acceptance: compute the deterministic handoff id
  (`ho:{analysis_id}:{outcome_hash16}` as design's `open_design_handoff` builds it);
  `HandoffStore.load` FIRST — absent → `HandoffGate.accept(manifest,
  "preparation-harness", {"approved"}, event_factory)`; recorded+accepted → re-verify
  entries against the manifest and proceed idempotently; recorded+rejected → typed
  terminal `failed` (`handoff_rejected`). Never call `record` twice (duplicate_handoff).
- The §4 nine entry checks over the four entries (TableSelection, ExperimentDesign,
  RunnableFrameContract, DeliveryCapacityCheck) resolved via `ProductsReader`:
  approved DesignOutcome lineage, hash chain, frame-contract↔design binding, CSV object
  hash equals TableSelection's resource_sha256, registry-version support (compare
  against the T-015 `PREPARATION_REGISTRY_KEYS` vocabulary), every eligibility/
  unusable/imputation rule id registered, pre-repair diagnostics present with approved
  handling, no upstream mutation marker, capacity check status `pass` with exact
  design/method/version binding. Fail closed with stable codes; collect ALL codes.
- `PreparationContextManifest` compiler: hydrate the T-015 model from the four entries
  plus the design-side MeasurementMap/RoleLedger/CausalContext reachable from
  ExperimentDesign; commit via the flush-gated committer path (caller-provided).

### 1.2 `src/causal/preparation/plancompile.py` (≤ 250 logical)
- Deterministic contract-gap computation: current frame schema vs the runnable-frame
  contract (missing prepared columns, type mismatches, sentinel evidence, required
  derivations, imputation targets) → stable gap codes.
- §7.4 grouping: `single` | `coupled` (shared output/key/derivation dependency) |
  `recipe` (shared fit scope) | `table_wide`; dependency edges; ≤8 concurrent groups
  with a deterministic queue; satisfied columns create NO task (EV-P3-001).
- §7.5 fan-in: reconcile `PreparationTaskDraftV1` proposals into ONE
  `PreparationPlanV1`: reject conflicting operations on a column, duplicate/inconsistent
  mappings, unregistered operations, protected-role targets, illegal fit scopes (per
  overlay), row-membership changes; require complete gap coverage; order via
  `has_cycle`; a draft carrying `DesignConflictDraftV1` routes to the conflict path.

### 1.3 `src/causal/preparation/validators.py` (≤ 220 logical)
The six §24.3 walls on `shared/validation.py` machinery, rules from
`registries/preparation-validation-rules.v1.json`:
1. entry/handoff; 2. rows (every source row exactly one terminal disposition; only
approved rule ids evaluated); 3. impact + method structure + freeze (counts reconcile,
support gates, `row_set_hash` present and stable); 4. plan (registered/permitted/
ordered/covering/protected/fit-scoped); 5. execution (every receipt `tri_agreement`
clean, input→output hash chain unbroken, row invariance); 6. final (required
diagnostics terminal with approved handling; runnable-frame contract satisfied;
PRD-004 §20 readability preconditions). Earlier walls never waived by later results.

### 1.4 Declarative
`registries/preparation-validation-rules.v1.json` (~120): one row per wall check with
stable codes, allowed actions, user_resolvable=false throughout (PRD-003 never asks).

### 1.5 Tests (≤ 380 logical)
Entry: accept/reject matrices (each of the nine checks), replay idempotency
(load-then-check), duplicate-handoff never raised. Plancompile: grouping table-driven
cases, cap + queue determinism, fan-in rejection matrix, coverage completeness, conflict
routing. Walls: table-driven pass+fail per wall using T-016/T-017 outputs as fixtures.

## 2. Constraints
Budgets: preparation +≤730 (→ ≈2,030/3,000 after T-016/17); modules +3; declarative
+≤130; tests ≤380 (headroom per D-073). No langgraph/model/CLI imports; design JSON
read as data only; `causal.design` imports FORBIDDEN except none. Pre-code projection;
one rethink.

## 3. T-016/T-017 absorption notes (committed models/APIs win over §1 wording)
- `eligibility_vocabulary` reader lives in `packs.py`; stabilization/impact APIs as
  committed at `f7ddf0a` (StabilizationRecordV1 assembly via `stabilization_summaries`).
- The executor consumes caller-supplied `expected_inputs={plan_item_id: ArtifactRef}` and
  an `OperationContext` (column roles, timing, permitted repair/imputation lists) — wall 4
  and the fan-in build these from the manifest; `PlanItemV1.parameters` are scalars only
  (lists/maps are canonical-JSON strings) — validate accordingly.
- Diagnostics registry rows carry `implementation: preparation_common|method_pack`;
  `method_pack` rows are dispatchable only via the executor's extra-impl hook (T-019) —
  wall 6 treats a registered-but-unimplemented method diagnostic as `not_computable`
  requiring approved handling, not as silent pass.
- Frame persistence: `shared/frames.py` (`FrameArtifactV1` with closed 6-dtype schema);
  executor Protocols `FrameWriter`/`FrameReader`/`ObjectWriter`.
- Test cap ≤380 is FIRM (tests 8,133/9,000; T-019/T-020 need the rest).
- Modules 62/68; per-module ≤350 logical; functions ≤75.
