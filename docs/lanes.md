# The lanes

A lane runs one family's analysis in its own process, on `designs/<n>/handoff.json` and the CSV, and reads nothing else. Three
are built: adjustment on DoWhy, diff-in-diff on pyfixest, discontinuity on rdrobust. They share a shape and never a method.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="diagrams/lane-swimlane-dark.svg">
  <img alt="The adjustment lane as a swimlane: fourteen stages in the code band, five judgements in the model band, three loops" src="diagrams/lane-swimlane-light.svg">
</picture>

## The harness

`causal_agent/lane/` is what every lane is built on. A lane's own judgements stay in its package; what every lane copies is here
once:

| module | what |
|---|---|
| [intake.py](../causal_agent/lane/intake.py) | the table by code: the CSV the pack names, the columns the pack names, the scope's row filter and window applied; a filter the small grammar cannot read is a `Decline`, never a guess |
| [case.py](../causal_agent/lane/case.py) | the pack weighed by code: a confirmed field is a fact the lane takes, a draft is open and may be asked of the model, a refuted or contradicted field is contested; the person's beliefs, unknowns and contradictions become flags by the lane's `beliefs.yaml`, and `decide_by_code` turns them into a stop, an ask, or a re-levelled check before any judgement |
| [verify.py](../causal_agent/lane/verify.py) | a model's answer about a column against what the pack settled: a claim that contradicts a fact is rejected unless it cites the contested address; every cite must resolve |
| [asks.py](../causal_agent/lane/asks.py) | one question back to the desk, the same shape in every lane: the lane stops with stage `ask`, the desk asks it, and runs the lane again |
| [figures.py](../causal_agent/lane/figures.py) | the figures a lane leaves, checked: every `draws_on` address must resolve in this run |
| [records.py](../causal_agent/lane/records.py) | `artifacts.json` with the common keys, the result the desk reads, the report's tail |
| [nodes.py](../causal_agent/lane/nodes.py), [prompts.py](../causal_agent/lane/prompts.py), [knowledge.py](../causal_agent/lane/knowledge.py) | the stream writer, the honest stop, the cards, the frame text every judgement reads first, the shared relate and pick and interpret prompts, the yaml loader |

The frame text ([lane/nodes.py:52](../causal_agent/lane/nodes.py)) is the one thing every judgement of a lane reads before its own
material: the decision the desk made, the pack's rendering of the dataset, the change, the beliefs, the family block, the design
brief and the person's words, the probes, and the case as code weighed it.

## The adjustment lane, stage by stage

[families/adjustment/lane/](../causal_agent/families/adjustment/lane/): `graph.py` wires it, `nodes.py` holds the stages,
`prompts.py` the five judgements, `adapter.py` is the only file that imports DoWhy.

1. **load** (code). Refuse what the lane cannot do: a question that is not the effect of a change, no treatment column, an outcome
   neither numeric nor two-valued, a dose with many values, an unsupported target. Otherwise the row count, the outcome kind, the
   observed levels.
2. **case** (code). Weigh the pack. Facts, drafts, open, contested, flags.
3. **contrast** (judgement, only if the pack does not name the treated level). Which levels are compared.
4. **relate** (judgement, one call per column the pack leaves open). A named instrument, a named mediator, a column the offer depended
   on and fixed before, or one the person called a measure of the outcome never reaches the model: `fact_relation` settles it
   ([nodes.py:286](../causal_agent/families/adjustment/lane/nodes.py)). Every other column gets four yes/no questions against the
   treatment and outcome cards, with what the pack already settled shown to be copied.
5. **merge_graph, verify_graph** (code). The DAG from the relations: a measure of the outcome is excluded, a post-treatment column
   that does not feed the treatment is excluded, a named mediator replaces the direct edge, a declared hidden factor becomes a node
   feeding both. Then the checks in [gates.md](gates.md); failing columns go back to relate with their errors, three rounds.
6. **identify** (code). DoWhy finds every road: back door, instrument, front door. If nothing identifies while a hidden factor
   stands, the lane asks the desk the one question that could open a road and stops. Once the person has said there is neither, the
   back door is taken with the hidden factor as a sensitivity range and a caveat. A brief that names a road must find it.
7. **check_design** (code). Per contrast: the smallest arm, common support, the score model's AUC, the standardised mean difference
   per adjustment column; thresholds from `checks.yaml` mark each soft or hard. The person's flags join as checks.
8. **assess** (yaml first, then a judgement, only when something flagged). Proceed, revise, or stop; a hard flag never permits proceed;
   a revision must touch a flagged column and loops to merge, three times at most.
9. **pick_estimator** (judgement). Code filters `estimators.yaml` by road, treatment type, outcome kind and adjustment set; the model
   picks one of the survivors by name, citing check addresses.
10. **freeze_design** (code). `design.json`: contrasts, graph, estimand, checks, estimator and its secondary, every refuter whose
    conditions match. Nothing loops after an estimate exists.
11. **analyse** (code, per contrast). DoWhy fits the primary, runs every matching refuter, fits the secondary. A failed fit excludes
    that estimator and picks once more.
12. **interpret** (judgement, per contrast). The answer from addressed material; every number must resolve within one percent.
13. **figures, assemble** (code). The graph, the balance, the effect against its refutations; the report and `result.json`.

Three yaml catalogues hold the knowledge, and none of it is a rule the model can bend:
[checks.yaml](../causal_agent/families/adjustment/lane/knowledge/checks.yaml) declares thresholds;
[estimators.yaml](../causal_agent/families/adjustment/lane/knowledge/estimators.yaml) is method knowledge the model chooses from after
code filters it, with parameters the model never sees;
[refuters.yaml](../causal_agent/families/adjustment/lane/knowledge/refuters.yaml) lists falsifications that all run when they apply;
[beliefs.yaml](../causal_agent/families/adjustment/lane/knowledge/beliefs.yaml) says what the person's beliefs mean to this lane.

## The other two, same shape, own method

| | diff-in-diff | discontinuity |
|---|---|---|
| engine | pyfixest | rdrobust and rddensity |
| stages | load, case, groups, periods, shape_table, relate, merge_controls, verify, check_design, assess, pick_estimator, freeze_design, estimate, placebo, interpret, figures, assemble | load, case, score, shape_table, relate, merge_covariates, verify, check_design, assess, pick_estimator, freeze_design, estimate, placebo, interpret, figures, assemble |
| its own judgements | groups, periods | score |
| the canonical shape code builds | `y, unit, time, treated, post, treat, rel_time, cohort` from a long or a wide table | `y, x, side` with the score recentred on the cutoff and the treated side positive |
| what the yaml declares | formulas over the canonical names, the inference rule, the placebos, the thresholds | the local polynomial specs, the inference rule, the placebo cutoffs and bandwidth grid, the thresholds |
| falsifications | placebo group, placebo timing | placebo cutoffs, the bandwidth grid, donuts |

Each is one package: `family.yaml`, `design.py`, `handoff.py`, `probes.py`, `postviz.py`, `lane/`, `evals/`, `tests/`, and one
line in [families/registry.py](../causal_agent/families/registry.py). The core never names a family;
[families/tests/test_core_names_no_family.py](../causal_agent/families/tests/test_core_names_no_family.py) greps for it.

## Run one

```bash
uv run python -m causal_agent.evals.lane adjustment students "Did completing the prep course raise math scores?"
```

```bash
uv run python -m causal_agent.evals.lane adjustment --handoff causal_agent/families/adjustment/evals/handoffs/gov_transfers_forced.json
```

The first routes the question through the desk's nodes without the interview, then runs the lane; the second runs the lane alone on
a stored pack. Both print the report and say where the run folder is.

Decided in [ADR 0001](adr/0001-lanes-stay-distinct.md), [ADR 0004](adr/0004-the-pack-is-the-only-input-a-lane-sees.md) and
[ADR 0005](adr/0005-one-folder-per-family.md).
