# T-019 — Preparation LangGraph harness, runtime dispatch, PRD-004 handoff

Status: frozen for implementation
Owning PRD: PRD-003 §7, §17.7, §18, §20, §22 (as §24); SC §4 (no interrupts), §7 rows,
§9; closes D-069 (resume + status gaps) and the D-071 echo-field cleanup
Depends on: T-015..T-018

## 1. Deliverables

### 1.1 `src/causal/preparation/graph.py` (≤ 290 logical)
- `PreparationDeps` frozen dataclass: conn, products, objects, committer, registry,
  emitter, clock, gateway, method packs + preparation overlay, task table, wall rules,
  prompts_root, frame writer/reader (shared/frames), repo_root.
- `StateGraph` over the §17.7 allowlist ONLY (ids, hashes, statuses, gap codes, counts,
  phase; never frames/payloads); `PostgresSaver` on its own connection, strict msgpack,
  no pickle (mirror `design/graph.build_checkpointer`); thread id
  `pt:{analysis_id}:{preparation_revision}`; **no interrupt nodes anywhere** (SC §4).
- `run_preparation(deps, *, analysis_id, design_outcome_artifact_id, stage_run_id)
  -> PreparationRunResult` (status, outcome ref, conflict ref, row_set_hash, handoff id);
  `preparation.preparation_runs` row upkeep (states per SC §4, terminal on every exit).
- `open_preparation_handoff(...)` — D-037 build-don't-record: PRD-004 manifest with the
  §20 four entries (PreparedFrameBundle, ExperimentDesign, RunnableFrameContract,
  DeliveryCapacityCheck), `receiving_stage_run_id = f"sr:{analysis_id}:estimation"`,
  readable only on outcome `prepared`.

### 1.2 `src/causal/preparation/nodes.py` (≤ 340 logical; split into two ≤350 modules only if forced)
Node sequence (§7 flow, lite): entry (T-018 `entry.py`, replay-safe) → manifest commit →
parse + row identity (T-016) → stabilization plan compile (T-018) → conditional Phase A
model task (ONLY on unresolved registered mismatch codes; **zero model calls on the
no-gap happy path**) → execute stabilization (T-016 engine) → impact + method structure →
freeze: commit `StabilizationRecord` + `StabilizedFrame` → post-freeze gap triage
(T-018) → conditional Phase B model tasks (shared `TaskRunner`, sequential within the
≤8 cap per D-050; hydrated context per §7.3 from the manifest — column roles, gap codes,
permitted operations, registered vocabularies in the prompt) → fan-in → `PreparationPlan`
commit → sequential mutation via T-017 executor (expected_inputs + OperationContext from
the manifest; method-pack diagnostics via the extra-impl hook) → per-step wall 5 →
diagnostics → wall 6 → `PreparedFrame` + `ExecutionReceiptBundle` + `PreparedFrameBundle`
commits → `PreparationOutcome` → handoff-open. `design_conflict` / `not_runnable` /
`failed` short-circuit to a terminal outcome with the conflict artifact committed.
Every boundary emits §18.6 events through the emitter; commits go through the
flush-gated committer.

### 1.3 Runtime + CLI integration (runtime ≤ +120; cli ≤ +30)
- `runtime/composition.py`: stage dispatch in `run()` — latest design run terminal
  `approved` with its recorded preparation handoff → run preparation against
  `--expected-stage-run pr-run id`; `status()` gains the preparation stage row AND
  (D-069b) prints the exact permitted next command for every non-terminal state.
- `runtime/failures.py` hardening (D-069/D-071): `guard` additionally catches broad
  `Exception` → stage run terminal `failed`, one `blocker.raised` with code
  `internal_error` (class name only, no message body), typed result exit 4 — no raw
  traceback can escape any command; `stale_revision` blockers name the expected id.
  Crashed-run resume (D-069a): a `running` row whose advisory lock is free is
  re-enterable — `run()` re-invokes the graph on the SAME thread id so the checkpoint
  resumes; a fresh attempt row is recorded.
- `cli/render.py`: render `PreparationOutcomeView` incl. conflict code + next command
  (re-run design as revision N+1). **No new CLI commands.**
- `shared/agenttask.py` (≤ +8): after the strict parse, the runner overwrites the
  model-echoed `envelope_id`, `task_id`, and `validation_target` with the harness-known
  values (D-071 — never depend on the model for facts the harness owns).

### 1.4 Declarative
`registries/preparation-tasks.v1.json` (~25): ONE task kind (`preparation_plan`) per
SC §5.4 — used by both phases with a `phase` context field; prompt path + version,
output schema, wall, budgets. `prompts/preparation/plan.v1.txt` (~85): mirrors the
design templates' discipline — JSON-only against the registered result schema, the
registered operation/diagnostic id vocabularies with usage cues, fit-scope rules,
`## allowed_evidence`/`## parent_artifacts` citation and echo rules, attempted-evidence
statuses, stopping states (`proposed`, `needs_dependency`, `design_conflict`, `failed`).

### 1.5 Tests (≤ 320 logical, FIRM — D-075 rebalance)
Parametrized scripted-gateway e2e over the four method packs: happy path (zero model
calls, `prepared` outcome, handoff manifest passes the T-006 gate) + one agent-gap path
(scripted Phase B proposal → plan → execution); restart at TWO boundaries (post-freeze,
post-plan-commit) across coordinator instances; `design_conflict` and `not_runnable`
terminals; broad-exception guard (a poisoned node → exit 4, run `failed`, one blocker);
crashed-run resume (kill mid-run simulation → re-run resumes from checkpoint); runtime
dispatch + status next-command + CLI render. Reuse the design e2e helpers to produce an
approved design fixture once per session (shared fixture), not per test.

## 2. Constraints
Budgets: preparation +≤640 (→ ≈2,700/3,000 incl. T-018); runtime ≤ 800 total; cli ≤ 500
total; shared ≤ 2,500; declarative +≤115; tests ≤320 (D-075); modules +≤3 (→ ≤68/68 — if a
nodes split is forced, something must consolidate first: flag instead of breaching).
No new CLI commands; no interrupts; PRD-003 never asks the user. Pre-code projection;
ONE rethink (this task is the wave's expected consumer). Where specs conflict with
committed models/APIs, the committed code wins — record deviations.

## 3. T-018 absorption notes (committed code wins over §1 wording)
- Entry gate APIs as committed at `ed1abcf`: `EntryPolicy` needs caller-supplied
  `imputation_strategy_ids`, `question_id`, `parser_profile_id` — the runtime composition
  supplies them (strategy ids from the preparation overlay; question id from the intake
  chain; parser profile is the pinned polars profile id constant).
- Wall 6 `handoff_readable` consumes a caller-supplied readiness map of PRD-003 §20's
  thirteen conditions — the outcome node builds it from real state before the bundle commit.
- Manifest commit goes through T-018's `ManifestCommitter` Protocol; pass the shared
  flush-gated `ArtifactCommitter`.
- Plan fan-in/table-wide/`__table__` semantics and scalar-only `PlanItemV1.parameters`
  as committed; wall 4 builds `OperationContext`/`expected_inputs` from the manifest.
- Tests 8,600/9,000: the ≤320 cap is FIRM; if the mandated coverage cannot fit, STOP and
  report the shortfall (blocked per §14.1.2) rather than breaching — the user decides.
- Modules 65/68: graph.py + nodes.py + (at most one more) = 68/68 exactly; a forced split
  beyond that must be flagged, not committed.
