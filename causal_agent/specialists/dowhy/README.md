# specialists/dowhy — the adjustment lane

Entered from the desk's hand-off for the `adjustment` family. Turns a question, a pack slice, and a table into a frozen Design, runs it on DoWhy, and writes a report or an honest Feasibility stop.

```
load ─ contrast ─(relate × N)─ merge_graph ─ verify_graph ─ identify ─ check_design
     ─(assess, only when flagged)─ pick_estimator ─ freeze_design ─(analyse × C)─ after_analyse
     ─(interpret × C)─ assemble
any typed stop ────────────────────────────────────────────────▶ feasibility ─ assemble
```

N = relevant columns other than treatment and outcome, C = contrasts. Neither appears in the graph.

## The roads

`identify` reads every road DoWhy finds on the graph: the back door, the front door, an instrument. The graph carries what the
pack settled as facts: a column the offer looked at and fixed before the change is a parent of both; the instrument the person
named points at the treatment alone; the mediator the person named carries the whole effect, so the direct edge goes; and when
the person says something outside the file drove both, a hidden node says so, which closes the back door. The estimator pick then
chooses among the estimators of the open roads, and the frozen design records the road taken. With no road open the lane asks the
desk one question (the mediator, then the instrument) and stops with status `ask`; the desk asks it like any other and runs again.
When the person has said there is neither, the lane takes the back door with the hidden factor left in as a sensitivity range, and
the reading says the effect holds only if that factor is no stronger than the simulated ones.

## The rule every node follows

- **Facts** are computed by code from declared inputs: `load`, `merge_graph`, `identify`, `check_design`, `freeze_design`, `analyse`, `assemble`. If an input is missing or an assumption fails they return a typed Feasibility stop, never a guess.
- **Judgements** call the model once, cite addresses, and pass a gate: `contrast`, `relate`, `assess`, `pick_estimator`, `interpret`. Five in total; two of them, the graph and the estimator, decide what runs.
- **Declarations** live in `knowledge/`: estimator catalogue, refuter catalogue, check thresholds. The model chooses names from the catalogue; parameters and thresholds are never asked of it.
- **Design freeze.** `freeze_design` writes the Design before any estimate exists. Loops happen only on failure facts (a rejected relation, a flagged check, a fit error). Nothing loops after an estimate.
- **The model never touches DoWhy.** `adapter.py` is the only file that imports it and it reads the Design.

## Files

- `contracts.py` — Relation, Graph, Estimand, Revision, DesignAssessment, EstimatorPick, Design. Lane-invariant artifacts (Contrast, Checks, Estimate, Refutation, Interpretation, Feasibility) are in `common/contracts.py`.
- `state.py` — SpecialistState; shares `question`, `handoff`, `dataset`, `specialist_result`, `debug` with the router.
- `prompts.py` — five prompts, method-free and column-free.
- `nodes.py` — the nodes; `graph.py` — the wiring; `adapter.py` — DoWhy calls; `checks.py` — overlap, separation, balance, arms.
- `knowledge/` — `estimators.yaml`, `refuters.yaml`, `checks.yaml` and their loader.
- `run.py` — CLI. `tests/` — fake model, real DoWhy on the real files. `evals/` — LangSmith dataset `causal-dowhy-v0`.

Runs write to `RUN_DIR` (default `.artifacts/runs/<dataset>-<id>/`): `table.csv`, `design.json`, `design.md`, `artifacts.json`, `report.md`.

## Run it

```bash
uv run python -m causal_agent.specialists.dowhy.run students "Did completing the prep course raise math scores?"
uv run python -m causal_agent.specialists.dowhy.run --handoff causal_agent/specialists/dowhy/evals/handoffs/gov_transfers_forced.json
uv run pytest causal_agent/specialists -q
```

## Known limits of this version

- Treatments are compared level against level. A dose (many numeric values) stops at `load`.
- Targets: average, on the treated, on the untreated. Conditional and counterfactual stop at `load`.
- Sensitivity to an unobserved confounder is not run. DoWhy's direct simulation was too slow on a thousand rows and its partial-R2 method needs a benchmark covariate; it returns as a refuter entry once wired.
- Refuter draws are seeded once per run by the adapter, not through DoWhy's `random_state`, which reseeds every simulation identically and turns the subset test into a guaranteed failure.
- Edges between covariates are not drawn. Backdoor identification on the star graph is valid without them.
- Clean data assumed: rows with a null in any relevant column are dropped at `load` and counted.
