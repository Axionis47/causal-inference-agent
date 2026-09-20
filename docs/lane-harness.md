# The three lanes on one harness: the pack weighed by code, the model for what is open, an ask-back, and a figure tail

Status, 19 Sept: every stage landed, one commit each: the harness (`causal_agent/lane/`), the figure contract and the graph kind, the three lanes on the harness with their beliefs files and ask-backs, the three figure tails, and the page showing the pair and the declines.

## Context

The desk now builds one memory per dataset and projects a pack (`common.contracts.Handoff`) that carries, for every field, how it
was settled (status, source, the person's sentence), the beliefs, the unknowns, the contradictions, the probes, and a family block.
The three lanes read that pack. An audit of each lane (19 Sept) shows they mostly render it into prompts and decide from the file
alone. The lanes stay distinct: adjustment on DoWhy, diff-in-diff on pyfixest, discontinuity on rdrobust. No lane takes on another's
design. What they share is the shape the adjustment lane already has, and this plan gives that shape to all three.

### What goes in today, and what does not

| pack content | adjustment (dowhy) | diff-in-diff (pyfixest) | discontinuity (rdrobust) |
|---|---|---|---|
| treated level, contrast | code | dead; control re-derived from the treatment column even when the group column differs | dead |
| design block: named columns | instrument, mediator, hidden node are edges by code | groups and periods are facts, same gate as the model; unit by code | score, cutoff, side, take-up, cluster, sampled-by-side by code |
| design block: allowed and forbidden lists | `forbidden` and `adjustment_candidates` rendered, never enforced | `controls_allowed` ignored; candidates from `relevant_columns` | `covariates_allowed` ignored |
| column fields `when`, `moved_by_change`, `measures_outcome` | `depends_on ∧ before` and `measures_outcome` are facts; `moved_by_change` re-asked to the model | all three re-asked to the model | all three re-asked to the model |
| provenance (status, source, said) | text on the card only | text only | text only |
| beliefs | `unobserved` is a node; `mediator`, `exclusion` open an ask by status only; `spillover` unused | `trend_continues`, `spillover`, `unobserved` reach the interpret prompt only; `DidDesign.trend_belief`/`spillover` not even rendered | `cutoff_only` never read nor rendered; `movable`, `score_fixed_before` rendered only |
| unknowns, contradictions | one prompt (relate), no branch | three prompts, no branch | three prompts, absent from interpret, no branch |
| probes | never rendered, never read | never read; `pre_periods` recomputed, never reconciled | never read; `rows_by_side` recomputed, never reconciled |
| scope filter and window | rendered, never applied to the table | rendered, never applied | rendered, never applied; `on_treated` silently answered with the cutoff effect |
| dataset, change, assignment cards | never reach a lane prompt | two prompts | one prompt |
| ask back | yes (mediator, exclusion), by status not value | none: a hard pre-trends flag ends as infeasible | none: a missing cutoff ends as infeasible |
| disagreement with the pack | `treated_level` dropped silently; pack columns outside `relevant_columns` never loaded | block rejection wipes the errors; cluster overridden; pre/post periods recomputed silently | block rejection wiped; cluster dropped silently; target substituted silently |
| figures after the run | one interval chart; no graph, no balance | one interval chart; event study only when the dynamic model ran; placebo draws discarded | one interval chart; `bins.csv` written and read by nothing |

### The flaw

The pack is a text the model may use, not a set of facts the code weighs. So a confirmed field and a drafted one look the same to
every gate; the person's beliefs never meet the test that bears on them (the trend belief never meets the pre-trends test, the
movable score never meets the density test); the lane re-asks the model what the person already said; when the lane overrides the
pack it does so silently; two of three lanes cannot ask back; and what the lane decided (the graph, the fitted paths, the jump) is
never drawn. This plan fixes each of those the same way in every lane.

### Principles that hold

- Facts from code, judgements from one gated model call each, declarations in YAML. Never patch code per dataset.
- The person's statement is evidence. In a lane it moves a flag level or opens an ask. It never writes a value.
- Every artifact has an address. A lane that overrides a pack field writes a `Decline` with an address, and the brief shows it.
- Nothing is drawn or stated that the run did not compute. Every figure's `draws_on` addresses must resolve or the figure is dropped.
- DoWhy stays in `specialists/dowhy/`, pyfixest in `did/adapter.py`, rdrobust in `rd/adapter.py`. The harness holds shape, not method.
- Prompt wording is load-bearing for the lane fakes (`for column '…'`, `NAMES YOU MAY PICK:`, `[estimate:<c>.value]`). Keep those
  anchors; new anchors (`SETTLED BY THE PACK`, `ADDRESSES YOU MUST CITE`) land with their fake in the same commit.

## 1. The shared harness: `causal_agent/lane/`

Code only, with tests. Every lane's `load`, `case`, `verify`, `assemble` and `figures` call into it.

### Contracts, in `common/contracts.py` beside `Feasibility`

```python
class Decline(BaseModel):
    """The lane did not take a pack field as given."""
    stage: str                                   # the node
    kind: Literal["declined", "replaced", "substituted"]
    about: str                                   # the pack address: scope.window, design.cluster_level, claim:assignment.cutoff, col:x
    pack_value: str | None = None
    took: str | None = None
    reason: str
    check: str                                   # the code rule: intake.filter_unparsed, groups.level_observed, shape.pre_periods
    cites: list[str] = []
    address -> f"decline:{stage}.{slug(about)}"

class LaneAsk(BaseModel):
    """One question back to the desk, keyed to a memory address the desk can settle."""
    address: str
    question: str
    options: list[str] = []
    because: str = ""                            # the check or belief that opened it, in words
    evidence: list[str] = []                     # check addresses shown beside the question
    stage: str = ""
```

Both go into the desk's serde allowlist (`desk/graph.py` `_CONTRACTS`).

### `lane/state.py`

`LaneState(TypedDict, total=False)` the three specialist states inherit: `handoff, dataset, run_dir, table_path, columns, declines:
Annotated[list[Decline], add], case, checks, check_facts: Annotated[dict, merge_dicts], ask: LaneAsk | None, feasibility, figures,
report`. Reducers `merge_dicts` and `by_key(keyfn)` (replace an item with the same key; used for `estimates` keyed `(contrast,
method)` and `refutations` keyed `(contrast, refuter)` so a re-pick does not duplicate). `RelateTask` gains `settled: str`.

### `lane/intake.py`: the table the lane works on

- `wanted_columns(h)`: treatment, outcome, `relevant_columns`, and every column the design block names (instrument, mediator,
  candidates; unit, time, group, controls allowed, cluster; score, take-up, cluster, covariates allowed). A pack-named column the
  frame forgot is loaded, not lost.
- `load(h, tag) -> Intake(table, columns, run_dir, table_path, declines, rows_before, rows_after)`. A design-named column missing
  from the file is a `Decline(check="intake.column_missing")`; a missing outcome or treatment stays a stop.
- `apply_filter(table, h.scope.population_filter)`: a small grammar by code (`col == v`, `!=`, `>=`, `<=`, `in [a, b]`, joined by
  `and`, columns matched by key). Unparsed or unknown column → `Decline(about="scope.population_filter",
  check="intake.filter_unparsed")`, table unchanged. Applied → `check_facts["intake"] = {filter, rows_before, rows_after}`.
- `apply_window(table, h.scope.window, time_key)`: `from A to B`, `A..B`, `>= A`, `<= A`, `after A`, `before A`; numbers or dates
  parsed like `did/shape._parse_like`. No time column → `Decline(check="intake.window_no_time_column")`.
- NA dropped on the wanted columns after the filter; `table.csv` written.

### `lane/case.py`: the pack weighed

- `Weight = fact | draft | open | contested`. `weigh_provenance`: confirmed by the person or a doc, or sourced `data`/`code:*` →
  fact; drafted → draft; refuted or contradiction → contested; unknown or empty → open.
- `settled(brief, field)` for `when`, `moved_by_change`, `measures_outcome`, `set_by`: the weight and the value.
- `Flag(name, level pass|soft|hard|stop, caveat, cites, ask: LaneAsk | None, combine: check name | None)`.
- `Case(facts: {address: value}, open: [address], flags: [Flag])` with `render()`: the block every lane prompt shows, headed
  `SETTLED BY THE PACK`, `OPEN`, `FLAGS`.
- `weigh(h, rules) -> Case` from the lane's `knowledge/beliefs.yaml`.
- `as_checks(case) -> [CheckResult(contrast="all", name="belief.<kind>" | "unknown.<addr>" | "contradiction.<addr>", level, detail)]`:
  a flag is a check. Assess, interpret, material and the page read it with no new code path.
- `decide_by_code(case, checks, rules) -> ("stop" | "ask" | "proceed", payload, checks')`: the yaml `with_check` table applied in
  order stop, ask, level edits. The assess judgement runs only if a flag remains after that.
- `already_asked(h, address)`: true when a `Said.about` in the pack names the address (the desk's `listen` records
  `about=", ".join(ask.addresses)`), so an ask opens once across the chain of designs.

### `knowledge/beliefs.yaml`, one per lane

```yaml
beliefs:
  trend_continues:                       # diff-in-diff
    value_field: believed
    by_status:
      confirmed_true:  {level: pass}
      confirmed_false: {level: hard, caveat: "the person says the treated group would have moved differently apart from the change"}
      drafted:         {level: soft, caveat: "…never confirmed", ask: {question: "…", options: [yes, no]}}
      unknown:         {level: soft, caveat: "the person could not say whether the groups would have kept moving together"}
      empty:           {level: soft, ask: {question: "…", options: [yes, no]}}
    with_check:
      pre_trends:
        - {belief: confirmed_false, check: hard, then: stop, reason: "the test and the person agree: the groups were already moving apart"}
        - {belief: confirmed_true, check: hard, then: ask, once: true, evidence: [check],
           question: "The groups were already moving apart before the change (p = {value}). You said: \"{said}\". Is there a reason they would have moved differently apart from the change?", options: [yes, no]}
        - {belief: confirmed_true, check: hard, asked: true, then: soften, caveat: "the person kept the belief after seeing the test; their reason: \"{said}\""}
  spillover:
    value_field: possible
    by_status:
      confirmed_true: {level: soft, caveat: "treated units could reach the comparison units, so the comparison carries part of the effect"}
unknowns:
  "claim:change.period_value": {level: hard, ask: {question: "…"}}
  "col:*.when": {level: soft, caveat: "when {column} was set is not known; it was used as the notes allow"}
contradictions:
  "col:*.when": {level: soft, caveat: "the file and the person disagree on when {column} was set; the lane took the file's reading [{check}]"}
```

`{value}`, `{said}`, `{column}`, `{check}` are filled by code. Status keys: `confirmed_true`, `confirmed_false`, `drafted`,
`unknown`, `contradiction`, `empty`. No dataset name may appear in any beliefs file (a test greps for it, as `rd/tests` does for
`checks.yaml`).

### `lane/verify.py`, `lane/asks.py`, `lane/records.py`, `lane/figures.py`

- `verify.contradictions(answer, brief, rules, h)`: a model claim that contradicts a pack fact (`affected_by_treatment=True` against
  `when=before`; `usable_as_control=True` against `moved_by_change=True`; `predetermined=True` against `when=after`) is rejected
  unless the answer cites that address and the address is in `h.contradictions`. `cites_resolve` shared.
- `asks.ask_back(stage, ask, reason) -> Command(goto="feasibility", update={feasibility: Feasibility(stage="ask"), ask})`;
  `asks.status_of(state)`: `ask` | `infeasible` | `done`.
- `records.artifacts(state, extra)` writes `artifacts.json` with the common keys `design, checks, check_facts, declines, case, ask,
  estimates, refutations, interpretations, feasibility, figures` plus the lane's own; `records.result(...)` the `specialist_result`
  with `status`, `ask`, `declines`, figure ids; `records.report_tail(state)`: a `DISAGREEMENTS WITH THE PACK` section, one line per
  decline, and an `ASKS BACK` line.
- `figures.ok_addresses(h, state)`: pack addresses ∪ `design.*` ∪ `check:*` ∪ `estimate:*` ∪ `refute:*`/`placebo:*` ∪ `decline:*`
  ∪ `probe:*`. `figures.write(run_dir, specs, ok)`: every spec through `viz.graph.check_spec`; a failing spec becomes a
  `Decline(stage="figures", check="figure.check")` and is dropped; `figures.json` written.

### Desk routing, `desk/nodes/journey.py`

- `after_run`: `ask_back` when the last run's status is `ask` and its `specialist_result["ask"]["address"]` is set, unless the
  run before it asked the same address (then `brief`, and the brief says the lane still asks it).
- `ask_back`: takes `options`, `because`, `evidence` from the `LaneAsk` when given (kind `choose` with options, else `open`), else
  derives options from the catalogue as today.
- `run` (stage 1): reads the lane's `figures.json` when it exists, else falls back to `postviz.figures(rec)`; prepends the
  ready-moment figure with `moment="ready"`.
- `desk/material.py`: decline lines `[decline:<stage>.<about>] kind · pack said X · lane took Y · reason (check)`; belief flags come
  through `checks` already. `brief`: one line "Where the lane disagreed with the pack: …" when any.

## 2. The figure tail (the contract, stage 1)

- `viz/spec.py`: `Kind` gains `graph`; `Node(id, label, role treatment|outcome|confounder|driver|mediator|instrument|hidden|excluded)`,
  `Edge(src, dst, cites)`; `FigureSpec` gains `nodes`, `edges`, `moment: ready | run`; `addresses()` adds `figure:<id>.node.<i>`
  and `.edge.<i>`; `render()` prints them.
- `viz/graph.py`: pure `check_spec(spec, ok)`: a graph needs nodes and edges whose ends are nodes; other kinds keep the current
  rules; `draws_on` must resolve in `ok`. The `check` node calls it with the memory's addresses.
- `viz/postviz/`: `common.py` with pure builders over dicts (`effect_and_refutations(estimates, refutations, contrast, prefix)`,
  `event_study(dynamic)`); `__init__.figures(rec)` stays as the desk fallback. Per-lane modules land with stages 5 to 7.
- Every lane gets a `figures` node (code) before `assemble`, on the happy path and on the feasibility path, that builds its specs
  from `check_facts`, the artifacts and the run dir's CSVs, and writes them through `lane.figures.write`.
- Web: `figure.ts` `Kind` += `graph`, `graphLayout(spec, box)` (treatment left, outcome right, mediator between, confounders and
  drivers above, hidden dashed); `Figure.tsx` draws circles, labels, arrows with an address in each edge's `<title>`;
  `RunDetail.tsx` shows the ready-moment figure beside the run's first figure under "before the run · what the run produced", then
  the rest.

## 3. Per lane

### Adjustment (`specialists/dowhy/`)

Node order after: `load → case → contrast → relate×N → merge_graph → verify_graph → identify → check_design → assess →
pick_estimator → freeze_design → analyse×C → after_analyse → interpret×C → figures → assemble`; `feasibility → figures → assemble`.

- `load` through `intake.load`; pack-named instrument, mediator and candidates load; the filter and the window apply by code;
  `intent != effect_of_change` is a stop that says so.
- `case` from `dowhy/knowledge/beliefs.yaml`: `unobserved` true → soft caveat (the node is the design's answer); `spillover` true
  or unknown → soft caveat; contradictions and unknowns on `col:*.when` → soft. `_frame_text` gains the dataset card, the change
  card, the probes, and `case.render()`; today none of the four reach a dowhy prompt.
- `contrast`: `h.control_level` used when set.
- `fact_relation` becomes `settled_claims(h, k)`: `affects_treatment` from `depends_on`; `affected_by_treatment` from
  `moved_by_change` when a fact, else from `when` (before → false, after → true); `is_outcome_measure` from `measures_outcome`;
  instrument and mediator as today. All four settled → no model call. Otherwise `relate` runs with a `SETTLED BY THE PACK` block and
  the settled values overwrite the model's after `verify_graph` checks the rest with `lane.verify`.
- `merge_graph`: `design.forbidden` enforced (a forbidden column is never a parent of the outcome; excluded with the pack cite);
  an adjustment set reaching outside a non-empty `adjustment_candidates` is a soft check `adjusts_outside_candidates`.
- `identify` ask-back by value and column: belief empty or drafted → ask `claim:<kind>.exists`; true with no column → ask
  `claim:<kind>.column`; true with a column not in the file → `Decline` and move to the next kind; unknown, contradiction or false →
  next. `sensitivity_required` set on every identify after `hidden_dropped`, so a revise loop keeps the sensitivity refuter.
- `check_design`: belief flags appended; `checks.run_checks` returns `check_facts["balance"] = {col: {before, after}}` (weighted SMD
  from the propensity it already fits) and `check_facts["propensity"]`.
- `assess`: `decide_by_code` first; the model only when flags remain.
- `interpret`: `ADDRESSES YOU MUST CITE` (every flagged check, including belief flags, and the interval); the gate enforces it.
- Reducers on `estimates`, `refutations`, `interpret_errors`, `declines`, `check_facts`.
- `assemble` through `lane.records`; `artifacts.json` gains the design, the graph, the checks, the declines, the case.
- Figures (stage 5, `viz/postviz/adjustment.py`): `causal_graph` (kind graph, roles from the graph, edges with their cites,
  excluded columns as nodes with no edges, `draws_on=["design.graph"] + cites`); `balance_<c>` (bars, before and after adjustment
  per column, an hline at the threshold, `draws_on` the balance check); `effect_<c>`.

### Diff-in-diff (`specialists/did/`)

Node order after: `load → case → groups → periods → shape_table → relate×N → merge_controls → verify → check_design → assess →
pick_estimator → freeze_design → estimate → placebo×K → interpret → figures → assemble`.

- `load` through intake; the window applies on `design.time` by code and lands in `Periods.window_start/end` so the design shows
  it; a window on a wide table is a `Decline`.
- `case` from `did/knowledge/beliefs.yaml` (the yaml above). `DidDesign.render()` prints `trend_belief` and `spillover`.
- `groups`, `periods`: a rejected block is a `Decline(kind="replaced", about="claim:assignment.treated_level" |
  "claim:change.period_value", check=…)` and the errors go into the model's first prompt instead of being wiped. A period value
  neither the block nor the model can settle is an ask on `claim:change.period_value`, not a stop.
- `shape_table`: the control level from the settled group column, not from `group_levels` keyed on the treatment column;
  `h.control_level` when the group column is the treatment column. Pack `pre_periods`, the `pre_periods` probe, and `staggered`
  reconciled with the shaped panel: a difference is a `Decline(kind="replaced", check="shape.pre_periods")`.
- `merge_controls`: `moved_by_change` fact → excluded without asking; `when=before` fact + `varies_over` as today; a non-empty
  `controls_allowed` honoured (a usable column outside it is excluded citing `design.controls_allowed`); an `add_control` revision on
  an absorbed column is rejected in the assess gate. `relate` only for open columns, with the settled block; `verify` with
  `lane.verify`.
- `check_design`: `_pre_trends` keeps the lead and lag coefficients in `check_facts["dynamic"]`, so the event study exists even
  when the run stops at assess; belief flags appended.
- `assess`: `decide_by_code`: trend false + hard pre-trends → stop by code; trend true + hard → ask once with the p and the said;
  asked and kept → soften to soft with the said in the caveat; spillover true → soft plus a control-group note. The model only when
  flags remain.
- `freeze_design`: `cluster_level` honoured when the column is in the panel and constant within unit (`inference.yaml` entries
  gain `cluster: unit` as the default; the adapter swaps in the design's cluster key), else a `Decline(check="inference.cluster_column")`.
- `estimate`: under `csw0` the primary is the advertised formula (all controls), the steps are secondaries. Say so in the commit body:
  the reported effect on cigar-like runs changes.
- `placebo`: `placebo_group` returns its 200 draws; `placebo_draws: Annotated[dict, merge_dicts]` into the artifacts.
- `interpret`: `ADDRESSES YOU MUST CITE`.
- Figures (stage 6, `viz/postviz/diff_in_diff.py`): `paths_<c>` (outcome by group over time with the treated group's path without
  the change: its pre-mean plus the control group's movement, post periods only, vline at the change); `event_study_<c>` (leads and
  lags, always when `dynamic` exists); `placebo_<c>` (density of the draws, vline at the observed effect, the p in the note);
  `effect_<c>`.

### Discontinuity (`specialists/rd/`)

Node order after: `load → case → score → shape_table → relate×N → merge_covariates → verify → check_design → assess →
pick_estimator → freeze_design → estimate → placebo×K → interpret → figures → assemble`.

- `load` through intake; the filter applies; the window only when a date column exists, else a `Decline`.
- `case` from `rd/knowledge/beliefs.yaml`: `cutoff_only` confirmed false → stop ("something else switches at the cutoff"); drafted
  or empty → ask `claim:cutoff_only.believed`; unknown → soft caveat. `movable` (address `claim:assignment.movable`) true →
  `with_check: density soft → harden` unless the score has a `set_by` fact; empty with a soft density flag → ask. `score_fixed_before`
  false → stop; empty → soft caveat on `col:<score>.when`. `RdDesign.render()` prints the `cutoff_only` belief.
- `score`: a rejected block is a `Decline(kind="replaced", about="claim:assignment.cutoff" | ".score_column" | ".treated_side")`
  and the errors go to the model. A known score with no cutoff → ask `claim:assignment.cutoff` ("Which value of '<score>' was the
  line drawn at?"); a cutoff rule with no score column → ask `claim:assignment.score_column`.
- `shape_table`: `scope.target == on_treated` → `Decline(kind="substituted", took="effect_at_cutoff", reason="a cutoff design
  estimates the effect for units at the cutoff; it has no average over the treated")`; `rows_by_side` probe reconciled with the
  shaped sides.
- `merge_covariates`: `when` fact → `predetermined` fact; `measures_outcome` fact; a non-empty `covariates_allowed` honoured;
  `relate` only for open columns; `verify` with `lane.verify`.
- `check_design`: belief flags appended; density hardened by code per the yaml.
- `assess`: `decide_by_code` first; the model as today with the argue-from-notes rule.
- `placebo`: every point kept (`placebo_points: {name: [{label, h, value, lo, hi, n_l, n_r, informative}]}`) into the artifacts.
- Figures (stage 7, `viz/postviz/discontinuity.py`): `rd_plot_<c>` (the `rdplot` bins from `bins.csv` as points, a fit on each side
  within the bandwidth from a triangular-kernel weighted polynomial of the design's order computed by code on a 40-point grid,
  vline at the cutoff); `density_<c>` (histogram either side, the rddensity p in the note); `continuity_<c>` (one interval per
  covariate, hline at 0); `bandwidths_<c>` (the estimate across the grid, vline at the chosen h); `placebo_cutoffs_<c>`; `effect_<c>`.

## 4. The desk shows it (stage 8)

- `server/models.py` `DeclineView`, `RunView.declines`; `server/sessions.py run_view` maps them.
- `web/src/types.ts`, `inspector/rows.ts` `declineRows`, `RunTables.tsx` `DeclinesTable`, `RunDetail.tsx` section "Where the lane
  disagreed with the pack" after the flags; the figure pair from stage 1 stays.
- `material.brief`: the run's figure ids in one line; `after.talk` unchanged (the run's first own figure by default; the
  ready-moment figure when the run asked back and made none).

## 5. Stages, one commit each, the app working after every one

| stage | commit | lands |
|---|---|---|
| 0 | The lane harness: intake, the case, declines, one ask shape | `causal_agent/lane/`, `Decline`/`LaneAsk`, desk routing for every lane, material and brief lines |
| 1 | Figures the lane leaves behind: the contract, the graph kind, the pair | `viz/spec.py` kinds, `check_spec`, postviz builders, `journey.run` reads `figures.json`, web graph kind and the pair |
| 2 | The adjustment lane on the harness | dowhy: intake, case, settled claims, forbidden enforced, ask by value, reducers, sensitivity kept |
| 3 | The diff-in-diff lane weighs the person's beliefs and asks back | did: intake, case, trend and spillover flags, controls allowed, cluster, primary formula, draws kept |
| 4 | The discontinuity lane weighs the person's beliefs and asks back | rd: intake, case, cutoff-only and movable flags, covariates allowed, target substitution recorded, points kept |
| 5 | The adjustment lane draws its graph and its balance | `postviz/adjustment.py`, dowhy `figures` node |
| 6 | The diff-in-diff lane draws paths, leads, and the placebo draws | `postviz/diff_in_diff.py`, did `figures` node |
| 7 | The discontinuity lane draws the jump, the density, and the bandwidth curve | `postviz/discontinuity.py`, rd `figures` node |
| 8 | The desk shows the pair and where the lane disagreed with the pack | server views, web tables, brief |

## 6. Tests

- Stage 0, `lane/tests/`: the filter grammar and the window on numbers and dates; an unparsed filter is a decline, not a stop; a
  design-named column loads outside `relevant_columns`; provenance → weight table; each `with_check` row of a fixture yaml
  (stop, ask, soften, caveat); `already_asked` through `Handoff.said`; a contradicting relate answer rejected without a cite and
  accepted with one. `desk/tests/test_journey.py`: an ask with options and evidence reaches the page; the same address asked twice
  reaches the brief; the brief lists a decline.
- Stage 1, `viz/tests`: graph addresses and render; `check_spec` rejects an edge to a missing node and an unresolved `draws_on`;
  the desk reads a lane's `figures.json` and falls back to postviz without one; `web`: `graphLayout` places treatment left of
  outcome; a graph spec renders its circles and arrows with an address per edge.
- Stage 2, dowhy: a forbidden column never enters the adjustment set; a pack-named mediator outside `relevant_columns` loads; a
  confirmed mediator with no column asks for the column; `sensitivity_required` survives a revise loop; a re-pick does not duplicate
  estimates; the filter and the window apply by code; spillover true is a caveat the interpretation must cite; relate is skipped when
  the pack settles all four claims.
- Stage 3, did: trend false + hard pre-trends stops with no assess call; trend true + hard asks back once with the p and the said,
  and with the address already in `said` softens instead; spillover caveat cited; `controls_allowed` honoured; a cluster not in the
  panel is declined with a record; the control level comes from the settled group column; pre-periods that differ from the pack are
  recorded; a rejected block is recorded and its errors shown to the model; the primary is the full formula under `csw0`; the window
  shrinks the panel; a missing period value asks back.
- Stage 4, rd: cutoff-only false stops by code; unasked cutoff-only asks back; movable true hardens a density flag unless the score
  has a `set_by` fact; movable empty with a density flag asks back; a missing cutoff asks instead of stopping; `on_treated` is
  substituted with a record; `covariates_allowed` honoured; a rejected block is recorded; placebo points are kept.
- Stages 5 to 7: each postviz module on a toy table; each lane's happy path writes its figure ids to `figures.json` with no
  `figure.check` decline; the rd fit's jump at the cutoff lies inside the primary's interval (a code sanity check).
- Stage 8: `run_view` maps declines; `declineRows`; the brief line.

## 7. Verification

- `uv run pytest causal_agent -q` after every stage; `cd web && npm test && npm run build` after stages 1 and 8.
- In the browser, with the server on :8000 and `LANGSMITH_TRACING=false`:
  students to ready, run, the inspector shows the overlap figure beside the causal graph, then the balance figure, no decline;
  cigar with the trend belief confirmed true, run, the desk asks back with the pre-trends p and the person's words; answer yes with a
  reason; the next run proceeds with a soft flag whose caveat quotes the reason; answer no; the next run stops by code and the
  report says the test and the person agree;
  senate, run, the RD plot, the density and the bandwidth curve appear; a forced pack with `target=on_treated` shows the
  substitution in the brief and the inspector.
- Review greps: no `dowhy` import outside `specialists/dowhy/`; no dataset name in any `beliefs.yaml`; every figure written passes
  `check_spec`.

## 8. Choices worth flagging

- A belief flag is a `CheckResult` named `belief.<kind>`. Assess, interpret, material and the page need no new path; `checks` now
  mixes data checks and belief flags, told apart by the prefix.
- "Soften a hard pre-trends flag after the person kept the belief with a reason" is the one place a person's word moves a level.
  It is declared in YAML, applied by code, the caveat quotes the said and cites the check, and it happens once per address.
- The `csw0` primary change alters which estimate the desk reports on runs with controls. The design already advertised the
  controls; the bare model was reported by accident.
- The old plan (memory, pack, desk, viz, stages 0 to 9) is complete and lives in `docs/desk-redesign.md`.
