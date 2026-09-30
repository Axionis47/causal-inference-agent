# families/adjustment: the DoWhy lane

Effect of a change on an outcome when the things that drove the change are measured in the data. The desk hands the lane a pack;
the lane turns it into a frozen design, runs it on DoWhy, and writes a report or an honest stop.

```
load ─ case ─ pair ─ mechanism ─ time ─ roles ─ post_roles ─ merge_graph ─ verify_graph ─ identify ─ check_design
     ─(assess, only when flagged)─ pick_estimator ─ freeze_design ─(analyse × C)─ after_analyse
     ─(interpret × C)─ figures ─ assemble
any typed stop, or a question back to the desk ─────────────────▶ feasibility ─ figures ─ assemble
```

C is the contrasts; it does not appear in the graph. `pair`, `mechanism`, `time`, `roles` and `post_roles` are the rungs of the
ladder (`lane/contracts.py`, `Ladder`): the pair, how the treatment was set, every column's place in time, the pre-treatment
columns placed together, the post-treatment columns placed together. A rung is code where the pack settles it and a bounded
episode where it does not, with the data tools and the budgets `knowledge/checks.yaml` declares. The stages are walked in
[docs/lanes.md](../../../docs/lanes.md); the judgements and their gates are in [docs/gates.md](../../../docs/gates.md).

## The roads

`identify` reads every road DoWhy finds on the graph: the back door, the front door, an instrument. The graph carries what the pack
settled as facts: a column the offer looked at and fixed before the change is a parent of both; the instrument the person named
points at the treatment alone; the mediator the person named carries the whole effect, so the direct edge goes; and when the person
says something outside the file drove both, a hidden node says so, which closes the back door. The estimator pick then chooses among
the estimators of the open roads, and the frozen design records the road taken. With no road open the lane asks the desk one
question, the mediator, then the instrument, and stops with status `ask`; the desk asks it like any other and runs again. When the
person has said there is neither, the lane takes the back door with the hidden factor left in as a sensitivity range, and the
reading says the effect holds only if that factor is no stronger than the simulated ones.

## The rule every node follows

- **Facts** are computed by code from declared inputs: `load`, `case`, `time`, `merge_graph`, `verify_graph`, `identify`,
  `check_design`, `freeze_design`, `analyse`, `figures`, `assemble`. A missing input or a failed assumption is a typed `Feasibility`
  stop, never a guess.
- **Judgements** are bounded episodes or single calls, cite addresses, and pass a gate: `pair`, `mechanism`, `roles`, `post_roles`
  (episodes, each only for what the pack leaves open), `assess`, `pick_estimator`, `interpret`. A rung cites the pack, a fact it
  asked the data for, or a rung below it; no tool joins the outcome with the treatment before the design is frozen.
- **The pack is weighed first.** `case` turns the pack into facts, drafts, open and contested fields, and the person's beliefs into
  flags by `knowledge/beliefs.yaml`; `decide_by_code` may stop or ask before any judgement. A column the pack settles never reaches
  the model.
- **Declarations live in `knowledge/`.** Thresholds in `checks.yaml`; the estimator catalogue in `estimators.yaml`, filtered by code
  before the model sees it, with parameters the model never sets; every refuter in `refuters.yaml` whose conditions match runs.
- **Design freeze.** `freeze_design` writes the design before any estimate exists. Loops happen only on failure facts: a rejected
  answer inside an episode, a flagged check, a fit error. Nothing loops after an estimate.
- **The model never touches DoWhy.** `lane/adapter.py` is the only file that imports it and it reads the design.

## Files

```
family.yaml     what it answers, needs, assumes; the decisions the Designer fills; needs_claims for the matrix
design.py       AdjustmentDesign, the family block in the pack
handoff.py      design_block(BlockInputs): how the desk fills the block, by code
probes.py       the arms and overlap probes; overlap.py the cell counts they use
postviz.py      the figures the lane draws from its own artifacts: the graph, the balance per contrast
lane/           graph.py, nodes.py, prompts.py, contracts.py (Relation, Graph, Estimand, Design, ...), state.py, adapter.py, checks.py
lane/knowledge  estimators.yaml, refuters.yaml, checks.yaml, beliefs.yaml
evals/          cases.yaml, handoffs/, the summariser, the evaluators, one EvalSpec
tests/          the lane's suite: a fake model, real DoWhy on the fixture files
```

A run leaves `table.csv`, `design.json`, `design.md`, `artifacts.json`, `figures.json` and `report.md` under `.artifacts/runs/<dataset>-dowhy-<id>/`.

## Run it

```bash
uv run python -m causal_agent.evals.lane adjustment students "Did completing the prep course raise math scores?"
```

```bash
uv run python -m causal_agent.evals.lane adjustment --handoff causal_agent/families/adjustment/evals/handoffs/gov_transfers_forced.json
```

```bash
uv run pytest causal_agent/families/adjustment -q
```

## Known limits

- Treatments are compared level against level. A dose with many numeric values stops at `load`.
- Targets: average, on the treated, on the untreated. Conditional and counterfactual stop at `load`.
- Refuter draws are seeded once per run by the adapter, not through DoWhy's `random_state`, which reseeds every simulation
  identically and turns the subset test into a guaranteed failure.
- Edges between covariates are not drawn. Back-door identification on the star graph is valid without them.
- Rows with a null in any relevant column are dropped at `load` and counted.
