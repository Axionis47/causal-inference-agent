# Analysis slice architecture

The analysis slice owns everything after a job reaches `CONFIRMED` at the
data-review gate. It picks up the parked `AnalysisState` (see
`backend/src/analysis_v2/PICKUP.md`), runs a controlled linear workflow of
agents, and ends with a verified notebook and a persisted, reopenable job.

This document is the build contract. `PICKUP.md` stays the input contract.

## 1. The linear spine

One workflow, forward by default. Stages are recorded on the run state, not
on the public `JobStatus`.

```
S0_DATASET_SAVED            (= input slice CONFIRMED; the pickup point)
S1_INTAKE_PARSED            IntakeAgent: question -> CausalSpec draft
S2_PROFILE_CREATED          ProfilingAgent: deterministic dataset profile
S3_DESIGN_CANDIDATES_CREATED DesignDetectionAgent: designs + tool eligibility
S4_TARGETED_EDA_COMPLETE    TargetedEDAAgent: base + design-specific EDA
S5_PLAN_CRITIQUED           PlanCriticAgent: gate the plan
S6_USER_CONFIRMED_OR_AUTO_APPROVED   human-in-loop or auto pass
S7_METHOD_EXECUTED          MethodLaneAgent: exactly one deterministic lane
S8_DIAGNOSTICS_SENSITIVITY_COMPLETE  DiagnosticsSensitivityAgent
S9_CLAIM_CRITIQUED          ClaimCriticAgent: claim strength + language
S10_REPORT_NOTEBOOK_CREATED ReportNotebookAgent
S11_NOTEBOOK_VERIFIED       NotebookVerificationAgent: execute + repair loop
S12_JOB_COMPLETE
```

Backward movement only on deterministic hard-gate failures (missing required
field, invalid column, notebook execution failure, analysis code failure,
user edits the spec, leakage / invalid variable role). Weak, fragile, or
inconclusive results are reported honestly and never trigger a rerun.

## 2. Where things live

```
backend/src/analysis_v2/
  state/         input-slice record (existing, untouched, schema v1)
  core/          run state, stages, events, artifacts, agent runs, gates
  spec/          typed cross-agent artifact schemas (CausalSpec, DesignCandidate,
                 EDAPlan, MethodPlan, EstimateResult, DiagnosticsResult,
                 SensitivityResult, ClaimCritique, Notebook*Result)
  persistence/   run-state records + artifact file IO
  runner/        LangGraph spine, gate router, progress maps, entry point
  agents/        one folder per agent (base/ holds the shared ReAct loop)
  tools/         deterministic tool functions (dataset, eda, methods, plots,
                 notebook builders)
  evals/         representative_cases.yaml, fixtures, agent + workflow evals
```

Rules carried over: 300 LOC hard cap per file (200 soft), tests co-located in
`<package>/tests/`, no sibling helper files, conventional commits.

## 3. Run state (single source of truth)

`AnalysisRunState` (new model, persisted as its own record kind; the
input-slice `AnalysisState` stays untouched at schema v1):

- identity: `job_id`, `schema_version`
- inputs snapshot: `causal_question`, `user_context`, `dataset_ref`
- slots, all typed, all nullable until their stage runs: `causal_spec`,
  `dataset_profile`, `design_candidates`, `selected_design`,
  `tool_eligibility`, `eda_plan`, `eda_summary`, `method_plan`,
  `estimate_result`, `diagnostics_result`, `sensitivity_result`,
  `claim_critique`, `final_report_path`, `notebook_path`,
  `notebook_verification`
- bookkeeping: `artifact_registry`, `agent_runs`, `state_events` (immutable
  append-only transitions), `current_state` (the S0..S12 stage),
  `state_version` (increments per commit), `status`
  (pending/running/waiting_for_user/failed/completed), timestamps,
  `total_tokens`, `total_cost_usd`

Agents never mutate this. Each agent returns a typed output; the runner
validates, commits the slot, registers artifacts, appends one `StateEvent`
(from_state, to_state, agent, input/output artifact ids, gate result,
warnings, tokens/cost, timestamp), bumps `state_version`, persists, then
moves on.

## 4. Public status and the existing wire

`JobStatus` (analysis_v2/state/status.py) gains three members the rest of
the stack already half-expects:

- `RUNNING = "running_analysis"` while the spine executes
- `WAITING_FOR_USER = "waiting_for_user"` parked at the S5/S6 plan gate
- `COMPLETED = "completed"` terminal; the literal string is already
  hardcoded in the SSE done-check, GET /results, and the storage terminal
  sets, so the enum member closes that gap

Touchpoints when the statuses land: manager orphan-recovery parked set
(add `waiting_for_user`), `_calculate_progress` / `_get_current_agent`
maps, and the frontend status display (unknown statuses currently render
as "live", which is acceptable until the analysis UI lands).

## 5. Orchestration

LangGraph `StateGraph` drives the spine. Nodes are agent stages; a dispatch
entry node routes to the next stage from `current_state`, which is also how
a `waiting_for_user` job resumes after confirmation. Conditional edges read
the stage's `GateResult`: `advance`, `park` (plan confirmation),
`fail`, or `back_to(stage)` for the enumerated hard failures only.

Workers are either deterministic tool-wrapped nodes or local ReAct loops
built on the existing `LLMClient` protocol (`generate_with_function_calling`
is single-shot; `agents/base` owns the caller-driven loop, max 3 repair
attempts, then a structured failure object). Agents receive only the tools
their stage and the current `tool_eligibility` allow.

Entry point: `JobManager.run_analysis` (manager.py:527) loads the parked
state and hands it to `analysis_v2.runner.entry.start(state, manager)`. The
runner re-registers the state in `manager._active_states` (the respawn
pattern) so SSE works, then drives the graph in the job task.

## 6. Gates

`GateResult{status, hard_failures[], soft_warnings[], reasons[]}`.

- Hard gates control progression: missing outcome column, treatment without
  variation, RDD without a confirmed cutoff, DID without time/group
  structure, notebook execution failure, method crash.
- Soft gates control interpretation only: weak pre-trends, poor overlap,
  fragile sensitivity, weak instrument, small near-cutoff sample, high
  missingness under the hard threshold. They downgrade confidence in S8/S9
  and never move the workflow backward.

Plan gate statuses at S5: `pass_auto_approved`, `needs_user_confirmation`
(park, emit confirmation card), `fail_missing_required_info`.

## 7. Artifacts

Every artifact is a file under `{LOCAL_STORAGE_PATH}/{job_id}/analysis/`
plus a registry entry `Artifact{artifact_id, kind, stage, agent, title,
path, media_type, summary, created_at}` on the run state. Kinds: json,
table (csv/parquet), plot (png), markdown, notebook, html. The registry is
what the frontend tiles list; `cleanup_local_artifacts` already removes the
whole job dir, so artifacts are cleaned for free.

Public reasoning summaries are artifacts too (markdown, one per agent run).
No private chain-of-thought is ever persisted or displayed.

## 8. Persistence

New record kind through the existing storage layer, both clients, following
the parked-state pattern: `save_analysis_run(state)`, `load_analysis_run
(job_id)`, `delete_analysis_run(job_id)` backed by `analysis_runs.json`
locally and an `analysis_runs` collection on Firestore. Serialization via
`dump_state_jsonable` (numpy-safe), versioned with `schema_version`.

The stale `save_results`/`save_traces` bodies (they read fields of the
deleted engine state) get rewritten when the report stage lands, so the
legacy GET /results and /traces routes keep working off the new run state.

## 9. SSE vocabulary (additive; the 7 dataset_* names are frozen)

- `analysis_stage_started` {stage, agent, headline}
- `analysis_agent_completed` {stage, agent, status, headline, warnings,
  artifact_ids, tokens, cost_usd}
- `analysis_artifact_emitted` {artifact_id, kind, title, stage}
- `analysis_gate_result` {stage, status, reasons, headline}
- `analysis_waiting_for_user` {confirmation_card}
- `analysis_completed` / `analysis_failed` {headline, ...}

Payloads carry a `headline` string so the existing terminal tape renders
them before the dedicated analysis UI exists.

## 10. New HTTP surface

- `GET /jobs/{id}/analysis` -> full run-state view (tiles, artifacts, costs,
  gate state); the reopen path for old jobs
- `GET /jobs/{id}/analysis/artifacts/{artifact_id}` -> artifact bytes
  (path-traversal safe, job-scoped)
- `POST /jobs/{id}/plan` -> confirm / edit the plan at the S6 gate
  (the existing /approval route stays data-gate only)
- existing `GET /results`, `/traces`, `/notebook` keep working

## 11. Evals

Three levels: schema/unit (state, registry, gates, eligibility),
agent-level (fixtures + expected outputs per agent), end-to-end workflow
runs over `evals/representative_cases.yaml`. Real datasets assert structure,
lane selection, direction, and known published benchmarks; synthetic
fixtures (DID panel, fuzzy RDD, IV, mediation, ITS step) assert magnitudes
against their generated ground truth and are labeled `synthetic` honestly.
No dataset-specific column names anywhere outside `evals/`.
