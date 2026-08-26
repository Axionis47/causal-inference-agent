# T-019 — Deterministic preparation coordinator, runtime dispatch, PRD-004 handoff (v2)

Status: frozen for implementation (v2 under PRD-003 Amendment 2 / D-076; supersedes v1 at
`f3072df`)
Owning PRD: PRD-003 §7, §17.7, §18, §20, §25 (V1-mech); SC §9 (coordinator note);
closes D-069 (resume + status gaps) and the D-071 echo-field cleanup
Depends on: T-015..T-018

## 0. Starting state

The working tree holds staged v1 work. It is the RAW MATERIAL for this task, not the
deliverable: the deterministic node bodies are kept; the model plumbing and LangGraph
shell are deleted. Nothing in the staged tree is committed; the implementer edits in
place.

## 1. Deliverables

### 1.1 Delete (Amendment 2 consequences)

- `prompts/preparation/plan.v1.txt` and `registries/preparation-tasks.v1.json` (drop
  from the index too).
- All `TaskRunner`/model-task plumbing: `HarnessBase._runner`, `run_task`,
  `task_context`, `_validate_draft`, `_exhausted`, `_task_row`, `_envelope`, the
  `TASK_ROW` SQL, `TASK_KIND`/`TASK_SCHEMA` constants, and `nodes._phase_a`.
- `src/causal/shared/agenttask.py` returns to its HEAD state PLUS at most +8 logical
  lines: after the strict parse, the runner overwrites model-echoed `envelope_id`,
  `task_id`, and `validation_target` with harness-known values (D-071). Nothing else
  from the staged +178 survives.
- LangGraph assembly: no `StateGraph`, `PostgresSaver`, or checkpointer in the
  preparation scope. `PreparationDeps` drops `gateway`, `checkpointer`, `task_table`,
  `prompts_root`; keeps the rest as staged.

### 1.2 `src/causal/preparation/plancompile.py` (+ ≤45 logical)

Deterministic gap→item compilation per PRD-003 §25.1: extend `auto_draft` (or a sibling
`compile_drafts`) so EVERY group compiles without a model —

- `required_derivation_missing` → `registered_derivation` (as committed);
- `imputation_target_missing` → operation by the registered strategy id
  (`numeric_median_with_indicator` → `numeric_median_imputation`;
  `categorical_explicit_missing_level` → `categorical_missing_encoding`), `fit_scope`
  copied from the pack target; a `cross_fit_training_fold` target compiles to the
  estimator-scoped recipe registration path already modeled in `reconcile`;
- `sentinel_evidence_present` → `missing_sentinel_normalization` with the evidenced
  mapping from the surface (the committed surface emits none today; the branch is
  written and fixture-tested);
- any other gap code → return a `ConflictRoute`-compatible `DesignConflictDraftV1`
  (stable code `no_registered_resolution`, the gap code and column in detail).

`DETERMINISTIC_GAPS` disappears as a concept: all gaps are deterministic or conflicts.
The fan-in `reconcile` and every wall check stay exactly as committed.

### 1.3 Coordinator + nodes (≤2 preparation modules, each ≤350 logical)

Target shape: `harness.py` (state dataclass/TypedDict, deps, `HarnessBase` minus model
plumbing) and `nodes.py` (node bodies + `run_preparation` + `open_preparation_handoff`);
`graph.py` is deleted or reduced into one of the two. Modules end ≤67 total (staged 66
minus deletions must not exceed 68).

- `run_preparation(deps, *, analysis_id, design_outcome_artifact_id, stage_run_id)
  -> PreparationRunResult`: plain sequential calls — entry → stabilize/freeze → plan →
  execute → outcome — each node returning the updated plain-dict state; a terminal
  `status` short-circuits to `outcome_node`. No interrupts. `preparation.
  preparation_runs` row upkeep per SC §4 (terminal on every exit path).
- Restart is artifact replay (D-035, PRD-003 §25.2): a rerun uses a NEW `stage_run_id`;
  deterministic artifact IDs make recommits no-ops; no committed-artifact skip logic.
- Keep the staged node bodies (entry/manifest, stabilize/freeze with walls 2–3, plan
  compile with wall 4, execute with wall 5, outcome with `_readiness` + wall 6 +
  handoff-open) with two dedups: ONE handoff-manifold builder shared by
  `open_preparation_handoff` and `_design_handoff`; ONE entry-parent reader replacing
  `entry_body`/`_entry_ref`/`_entry_hash`.
- `stabilize_node`: `unresolved_conflict` rows go straight to
  `self.conflict(state, pc.conflict_draft(UNRESOLVED, rows))` (no Phase A).
- `plan_node`: every group goes through the §1.2 deterministic compilation; a conflict
  draft routes through `self.conflict`.
- Every boundary emits its §18.6 events (agent rows vacated); commits go through the
  flush-gated committer.

### 1.4 Runtime + CLI integration (runtime ≤ +120; cli ≤ +30)

- `runtime/composition.py`: stage dispatch in `run()` — latest design run terminal
  `approved` with its recorded preparation handoff → run preparation; `status()` gains
  the preparation stage row AND (D-069b) prints the exact permitted next command for
  every non-terminal state.
- `runtime/failures.py` (D-069/D-071): `guard` additionally catches broad `Exception` →
  stage run terminal `failed`, one `blocker.raised` with code `internal_error` (class
  name only), typed result exit 4; `stale_revision` blockers name the expected id.
  Crashed-run handling (D-069a): a `running` preparation row whose advisory lock is
  free is re-enterable — `run()` starts a fresh attempt via artifact replay (new
  `stage_run_id`; no checkpoint).
- `cli/render.py`: render `PreparationOutcomeView` incl. conflict code + next command
  (re-run design as revision N+1). No new CLI commands.

## 2. Tests (≤ 320 logical, FIRM — D-075 rebalance)

Parametrized scripted e2e over the four method packs: happy path (ZERO model calls —
assert no gateway construction — `prepared` outcome, handoff manifest passes the T-006
gate) + one deterministic-gap path (imputation gap → compiled plan → execution →
`prepared`); rerun idempotency (run twice; second run commits no duplicate artifacts and
reaches the same terminal state — this replaces v1's checkpoint-restart legs);
`design_conflict` (unresolved rows AND an uncompilable gap) and `not_runnable`
terminals; broad-exception guard (poisoned node → exit 4, run `failed`, one blocker);
runtime dispatch + status next-command + CLI render. Reuse the design e2e helpers to
produce an approved design fixture once per session (shared fixture), not per test.

## 3. Constraints

Budgets: preparation ends ≤3,000 total (projection ≈2,550–2,700); runtime ≤800 total;
cli ≤500 total; shared ≤2,500 (agenttask reverts per §1.1); declarative delta ≤0
(deletions only); tests ≤320 FIRM — a genuine shortfall BLOCKS per SC §14.1.2 and goes
to the user; modules ≤68 with every module ≤350. The wave's ONE rethink was consumed by
this v2 revision (D-076); a further breach is terminal and goes to the user. Where this
spec conflicts with committed models/APIs, the committed code wins — record deviations.
No new CLI commands; no interrupts; PRD-003 never asks the user; no model calls
anywhere in the preparation scope.
