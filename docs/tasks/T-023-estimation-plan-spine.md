# T-023 — Estimation plan spine: entry conditions, plan compiler, capacity recheck, 15 walls

Status: frozen for implementation
Owning PRD: PRD-004 §4, §6.1, §6.5, §18, §26
Depends on: T-022

## 1. Deliverables

1. `src/causal/estimation/plancompile.py` (≤ 280 logical):
   - `compile_context_manifest(...)` — builds `EstimationContextManifestV1` from the
     four §4 handoff artifacts (prepared bundle, design, frame contract, design-time
     capacity check) after verifying the twelve §4 entry conditions; each failure is
     a stable code feeding wall 1. Mirrors `preparation/entry.py` structure.
   - `compile_plan(manifest, pack) -> EstimationPlanV1` — deterministic §6.1
     compilation from approved artifacts + pack registry; no choice points; seed
     derivation via shared `derive_seed` (31-bit, D-061) from plan identity fields.
   - `recheck_capacity(plan, prepared_structure) -> pass | DesignConflict draft` —
     §6.5 exact recheck reading `registries/delivery-capacity.v1.json` and the
     visualization-catalog version bound at approval DIRECTLY (never importing
     `causal.design`); verifies frozen result cardinalities, evidence families,
     templates, budgets against the design-approval check's bound versions (~45
     lines; the design harness's capacity preflight is the reference semantics).
2. `src/causal/estimation/walls.py` (≤ 220): the fifteen §18 walls in the
   `preparation/validators.py` shape — a `WallContext` dataclass + predicate table
   dispatched by rule kind over `registries/estimation-validation-rules.v1.json`
   (~22 rules, ~130 declarative). Walls 1-5, 8-12, 14-15 are thin declarative
   predicates; wall 6 (fold leakage: fold-fit receipts reference training rows only,
   from CrossFitAssignment counts), wall 7 (atomic primary_items vs plan contrast
   set), wall 13 (delegates to T-025's claim validator) are bespoke. No later wall
   waives an earlier failure; reports via shared `ValidationReport`.
3. `src/causal/shared/validation.py` — ONE line: `ValidationReport.wall` bound
   `le=10` → `le=15` (declared by PRD-004 §26.2).

## 2. Tests (≤ 350 logical)

Manifest compilation: golden path from a T-019-shaped prepared handoff fixture +
each of the twelve entry conditions individually violated → its stable code. Plan
determinism: same inputs → identical plan hash; any field change → new hash.
Capacity recheck: pass and conflict paths. Walls: fixture WallContexts driving every
wall to pass and to its characteristic failure; ordering (early failure blocks later
walls); leakage wall catches a poisoned fold fixture.

## 3. Constraints

estimation ≤ 960 total after this task (+500); shared +1 only; declarative ≤ +130;
tests ≤ 350; modules +2. Wave rethink shared. Committed code wins; record deviations.
