# analysis_v2

The analysis slice: everything after a job reaches CONFIRMED at the
data-review gate. `PICKUP.md` is the input contract (where every field the
analysis can read lives on the picked-up state).
`docs/analysis-slice/00-architecture.md` is the build contract: the linear
S0..S12 spine, the run state, gates, artifacts, persistence, and wire.

Structure:

  state/        the input-slice record this slice picks up (schema v1, sealed)
  core/         run state, stages, events, artifacts, agent runs, gates
  spec/         typed cross-agent artifact schemas (CausalSpec, designs, results)
  persistence/  run-state records + artifact file IO
  runner/       LangGraph spine, gate router, entry from JobManager.run_analysis
  agents/       one folder per agent; base/ holds the shared ReAct loop
  tools/        deterministic tool functions (dataset, eda, methods, plots)
  evals/        representative_cases.yaml + fixtures + agent/workflow evals

Built one agent at a time; every milestone lands with co-located tests.
