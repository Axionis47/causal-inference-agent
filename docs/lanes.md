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
| [nodes.py](../causal_agent/lane/nodes.py), [prompts.py](../causal_agent/lane/prompts.py), [knowledge.py](../causal_agent/lane/knowledge.py) | the stream writer, the honest stop, the cards, the frame text every judgement reads first (the pack, the case, the ladder so far), the data tools over the run's table, the shared pick and interpret prompts, the yaml loader |

The frame text ([lane/nodes.py:52](../causal_agent/lane/nodes.py)) is the one thing every judgement of a lane reads before its own
material: the decision the desk made, the pack's rendering of the dataset, the change, the beliefs, the family block, the design
brief and the person's words, the probes, and the case as code weighed it.

## The adjustment lane, stage by stage

[families/adjustment/lane/](../causal_agent/families/adjustment/lane/): `graph.py` wires it, `nodes.py` holds the stages,
`prompts.py` the judgements, `adapter.py` is the only file that imports DoWhy. The design is climbed as a ladder, in the order an
analyst reads the problem: the pair, the mechanism, time, the columns fixed before the change all together, the columns set at or
after it all together, the graph, the road, where the effect could differ, and the threats. Each rung is code where the pack
settles it and a bounded episode where it does not
([lane/episode.py](../causal_agent/lane/episode.py)): the model may look at the data through six read-only tools
([lane/tools.py](../causal_agent/lane/tools.py)), every fact it asks for comes back as an addressed line (`probe:<rung>.<n>`), and the
answer is one typed record that code gates, three tries inside the episode. No tool joins the outcome with the treatment before the
design is frozen; that refusal is code, and the model reads it. Every rung's lines (`ladder:<rung>.<field>`) are read by the rungs
above it, by the interpretation, by the report and by the chat after.

1. **load** (code). Refuse what the lane cannot do: a question that is not the effect of a change, no treatment column, an outcome
   neither numeric nor two-valued, a dose with many values, an unsupported target. Otherwise the row count, the outcome kind, the
   observed levels.
2. **case** (code). Weigh the pack. Facts, drafts, open, contested, flags.
3. **pair** (rung 0; code, an episode only if the pack does not name the treated level). The outcome, the treatment, which levels
   are compared, and the target the question asked for.
4. **mechanism** (rung 1; code from the pack's assignment claims, an episode when the pack names no drivers). The kind of
   assignment, the columns the decision or the offer looked at, an offer column and an uptake column when they are two, and whether
   units could move their own assignment. The episode's drivers must be columns in play, and a lottery leaves no room for choice.
5. **time** (rung 2; code). Every other column before, at, after the change, or unknown, from the pack's `when` fields.
6. **roles** (rung 3; one episode over every column fixed before the change, or of unknown timing, that the pack leaves open). A
   named instrument, a named mediator, a column the offer depended on and fixed before, or one the person called a measure of the
   outcome never reaches the model: `fact_relation` settles it. Every other column is placed together with the rest: the four claims
   against the pair, what it stands for, whether it carries the same information as another column or sits inside one, and
   whether it is a candidate modifier. A redundancy or a nesting is accepted only on a fact: a redundancy tool result, a
   `probe:data.redundancy.*` line, or the column's own `same_as` or `nested_in`. The gate copies what the pack settled, refuses a
   contradiction with a pack fact, and refuses a departure from the last run's reading that names no `Departure` with a cite.
7. **post_roles** (rung 4; one episode over every column set at or after the change that the pack leaves open). Each is a mediator,
   another measure of the outcome, a consequence of the treatment, a consequence of the outcome, a background attribute recorded
   late, or unrelated. None is adjusted for. The outcome by arm is refused to this rung, so no column is placed by peeking at the effect.
8. **merge_graph, verify_graph** (code). The DAG from the roles: a measure of the outcome is excluded, a post-treatment column that
   does not feed the treatment is excluded, a named mediator replaces the direct edge, a declared hidden factor becomes a node feeding
   both; of two columns that carry one thing the graph keeps one, the finer of a nested pair, and says why. Then the graph as a whole:
   acyclic, every node a column, a role for every column; a graph that fails is an honest stop.
9. **road** (rung 6; code, an episode only when more than one road is open). DoWhy finds every road: back door, instrument, front
   door. If nothing identifies while a hidden factor stands, the lane asks the desk the one question that could open a road and
   stops; that is the only question a lane asks, because it depends on the lane's own graph. Once the person has said there is
   neither, the back door is taken with the hidden factor as a sensitivity range and a caveat. A brief that names a road must find
   it. With one road open, code takes it; with more, a judgement argues which from the hidden factors, the mechanism and the roles,
   and the estimator pick then offers only that road's estimators.
10. **heterogeneity** (rung 7; an episode over the candidates, the target by code). The candidates are the pre-treatment columns
    the roles rung, or the person, marked as ones the effect could differ by. The judgement names at most three, each cited, for a
    reason in the story; the outcome by arm is refused to it. The target follows from the mechanism and the question: what the
    question asked when it asked; else the effect on the treated when units chose after an offer; else the average.
11. **threats** (rung 8; code). The risks of this design that no check measures, each from a pack line or a rung below: selection
    into the file, the outcome's timing, an offer that differs from the taking, choice with the hidden factors unsettled, gaps whose
    reason is not settled. Each is a flag the assessment must answer and the interpretation must cite. Beside them, every item a
    rung would not guess (`unsure`) is a soft flag; when it names a claim the interview could have settled, it is also a decline,
    `needs.unsettled`, which says a decision in `family.yaml` should rest on it so the next interview asks first.
12. **check_design** (code). Per contrast: the smallest arm, common support, the score model's AUC, the standardised mean difference
    per adjustment column; thresholds from `checks.yaml` mark each soft or hard. The person's flags, the threats and the unsure
    items join as checks.
13. **assess** (yaml first, then a judgement, only when something flagged). Proceed, revise, or stop; a hard flag never permits proceed;
   a revision must touch a flagged column and loops to merge, three times at most.
14. **pick_estimator** (judgement). Code filters `estimators.yaml` by road, treatment type, outcome kind and adjustment set; the model
   picks one of the survivors by name, citing check addresses.
15. **freeze_design** (code). `design.json`: contrasts, graph, estimand, checks, estimator and its secondary, every refuter whose
    conditions match. Nothing loops after an estimate exists.
16. **analyse** (code, per contrast). DoWhy fits the primary, runs every matching refuter, fits the secondary, then the primary
    again within each level of each modifier, on the graph without that column; too few rows in an arm is recorded, never skipped
    in silence. A failed primary fit excludes that estimator and picks once more.
17. **interpret** (judgement, per contrast). The answer from addressed material, the ladder's lines, the episodes' facts and the
    effect within each modifier's levels among it; every flag and every within-level estimate must be cited; every number must
    resolve within one percent.
18. **figures, assemble** (code). The graph, the balance, the effect against its refutations, the effect within each modifier's
    levels; the report with the ladder and what each episode looked at, and `result.json`.

Three yaml catalogues hold the knowledge, and none of it is a rule the model can bend:
[checks.yaml](../causal_agent/families/adjustment/lane/knowledge/checks.yaml) declares thresholds;
[estimators.yaml](../causal_agent/families/adjustment/lane/knowledge/estimators.yaml) is method knowledge the model chooses from after
code filters it, with parameters the model never sees;
[refuters.yaml](../causal_agent/families/adjustment/lane/knowledge/refuters.yaml) lists falsifications that all run when they apply;
[beliefs.yaml](../causal_agent/families/adjustment/lane/knowledge/beliefs.yaml) says what the person's beliefs mean to this lane.

## The diff-in-diff lane, rung by rung

[families/diff_in_diff/lane/](../causal_agent/families/diff_in_diff/lane/) climbs its own ladder on the same harness, in the order
an analyst reads a before-and-after comparison. Every rung is code where the pack settles it and a bounded episode where it does
not, with the budgets its `checks.yaml` declares; every claim cites the pack, a fact the episode asked for, or a rung below.

1. **groups** (rung 0). Who got the change: the column and the level; the pack's block when it names one and the file bears it out,
   else a judgement, with the pack's failed answer shown as the first rejection.
2. **periods** (rung 1). The clock: a time column and the first period at or after the change, or a before and an after column.
   A time column the file has but a first period nobody can settle is the one question this lane asks back.
3. **shape_table** (rung 2; code). The canonical panel and its facts: units per group, periods before and after, cohorts.
4. **comparison** (rung 3; an episode). Whether the comparison group is a fair stand-in for the treated group without the change,
   argued from the story and the facts, and every risk the story raises: anticipation, spillover, composition, another shock,
   the group chosen for where its outcome was heading. Each risk is cited and becomes a flag. The outcome by group is refused; the
   test of the paths before the change is code, later.
5. **controls** (rung 4; one episode over every candidate the pack leaves open). Changed by the treatment, usable as a control, or a
   unit trait the effect could differ by; what the pack settled is copied. `merge_controls` and `verify` then build the control set
   by code: a column fixed within a unit or within a period is absorbed by the fixed effects whatever the model said.
6. **heterogeneity** (rung 5). The candidates are the unit traits, read off the panel, that the controls rung or the person marked;
   a judgement names at most three; the primary estimator runs again within each level of each, without that column among the
   controls. The target is the effect on the treated, which is what this design identifies.
7. **threats** (rung 6; code). The risks every design carries from the pack, and the risks the comparison rung named. Each is a flag
   the assessment answers and the interpretation cites; an `unsure` on a claim the interview could have settled is a decline.
8. Then as before: `check_design` (the pre-trends test among the checks), `assess`, `pick_estimator`, `freeze_design` (which adds the
   clustering rung, `ladder:cluster.*`, by code), `estimate`, the placebos, `interpret`, `figures`, `assemble`.

## The discontinuity lane, rung by rung

[families/discontinuity/lane/](../causal_agent/families/discontinuity/lane/) climbs its own ladder on the same harness, in the order
an analyst reads a cutoff design.

1. **score** (rung 0). The score, the cutoff, which side got the change, and who took it up; the pack's block when it names one
   and the file bears it out, else a judgement. A rule the notes state but leave incomplete is the one question this lane asks back.
2. **shape_table** (rung 1; code). The canonical table: the score recentred on the cutoff, the rows on each side, sharp or fuzzy.
3. **density** (evidence; code). Before the line is judged: the density test at the line with the settings `checks.yaml` declares,
   the split of the rows in nested windows either side (a coin toss under no manipulation), the histogram, the mass points, or the
   word that the rows were drawn by side and the test says nothing. Each is a line of the ladder (`ladder:density.*`), computed once;
   the check and the figure read the rung.
4. **line** (rung 2; an episode). Whether the line is clean, argued from the story and the evidence: what set the score, whether a
   unit could move it, what else switches there; every risk the story raises, from a fixed list (manipulation, another change at the
   line, a score set after the decision, a cutoff known in advance), each cited and each a flag. A clean verdict over bunching must
   cite the density line; bunching with the person's word that the score could be moved must name the manipulation risk. The
   outcome by side is refused; the rung may look at a column by side within a band and at the score's histogram.
5. **covariates** (rung 3; one episode over every candidate the pack leaves open). Fixed before the line, changed by the treatment,
   another measure of the outcome, or a predetermined characteristic the effect could differ by; what the pack settled is copied.
   `merge_covariates` and `verify` then build the set by code.
6. **heterogeneity** (rung 4). The candidates are the predetermined characteristics the covariates rung or the person marked; a
   judgement names at most three; the primary spec is fitted again within each level of each, at the design's bandwidth so the
   levels compare, without that column among the covariates.
7. **threats** (rung 5; code). The risks every design carries from the pack, and the risks the line rung named. Each is a flag the
   assessment answers and the interpretation cites; an `unsure` on a claim the interview could have settled is a decline.
8. Then as before: `check_design` (the density flag read off the rung, and the covariate continuity, among the checks), `assess`, `pick_estimator`,
   `freeze_design` (which adds the window rung, `ladder:bandwidth.*`, by code), `estimate`, the falsifications, `interpret`,
   `figures`, `assemble`.

## The three lanes side by side

| | diff-in-diff | discontinuity |
|---|---|---|
| engine | pyfixest | rdrobust and rddensity |
| stages | load, case, groups, periods, shape_table, comparison, controls, merge_controls, verify, heterogeneity, threats, check_design, assess, pick_estimator, freeze_design, estimate, placebo, interpret, figures, assemble | load, case, score, shape_table, line, covariates, merge_covariates, verify, heterogeneity, threats, check_design, assess, pick_estimator, freeze_design, estimate, placebo, interpret, figures, assemble |
| its own judgements | groups, periods, comparison, controls, heterogeneity | score, line, covariates, heterogeneity |
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
