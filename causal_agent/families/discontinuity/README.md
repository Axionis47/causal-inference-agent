# families/discontinuity: the rdrobust lane

Effect of a change that a line on a measured score decided. The desk hands the lane a pack; the lane builds a canonical cutoff
table, freezes a design, fits it on rdrobust, runs the falsifications, and writes a report or an honest stop.

```
load ─ case ─ score ─ shape_table ─ line ─ covariates ─ merge_covariates ─ verify ─ heterogeneity ─ threats ─ check_design
     ─(assess, only when flagged)─ pick_estimator ─ freeze_design ─ estimate ─(placebo × K)─ interpret ─ figures ─ assemble
any typed stop, or a question back to the desk ────────────────────────────▶ feasibility ─ figures ─ assemble
```

K is the placebos the catalogue declares. `score`, `shape_table`, `line`, `covariates`, `heterogeneity` and `threats` are the rungs
of the ladder (`lane/contracts.py`, `Ladder`), and `freeze_design` adds the window rung by code: the score and the line, the shape
of the two sides, whether the line is clean and what could break it, the candidate covariates placed together, where the effect
at the cutoff could differ, the risks, how wide the window is. A rung is code where the pack settles it and a bounded episode where
it does not, with the data tools and the budgets `knowledge/checks.yaml` declares.

## How it sits on rdrobust and rddensity

- **Canonical inputs.** `lane/shape.py` produces `y, x, side`, plus `t` when a column records receipt, `cluster` when the pack declares
  an entity, and every numeric candidate covariate. `x` is the score recentred on the cutoff and flipped so the treated side is
  positive, so every fit, check and placebo is written for one geometry. When the notes say a unit exactly at the cutoff is control,
  the effective cutoff moves to the midpoint before the next distinct treated score, and the shift is a recorded fact.
- **Local polynomial spec.** `knowledge/estimators.yaml` holds named specs with the standard defaults pinned: local linear,
  triangular kernel, mass points adjusted. How far from the line the fit reaches is the window rung's judgement, not a parameter of
  the spec.
- **The window.** After the estimator is picked, code builds the table of every width selector the library offers
  (`checks.yaml window.selectors`), the width each gives on each side and the rows it leaves inside; a judgement picks one by
  name, the default (`mserd`) unless the density, the balance or the sides argue for another, and the gate holds the floor on
  rows a side and the citation a departure needs. When the score has fewer distinct values than `checks.yaml` declares there is
  nothing to judge: `h` keeps three support points a side and the design says so. The primary fits in the chosen window on both
  sides; a refit on other rows re-selects with the same selector, or keeps the pinned window when the rule has no selector.
- **Local randomisation.** For a score with few distinct values (below `checks.yaml support.distinct_min`) the catalogue offers
  `local_randomisation` on rdlocrand: the window rung takes the largest window in which the predetermined covariates stay
  balanced (or the support-points window when there are none), the estimate is the difference in means inside it (the
  Anderson-Rubin statistic and the Wald ratio when take-up is partial), the p-value comes from reshuffling the sides and the
  interval from inverting that test over a grid. Its own sensitivities run instead of the polynomial falsifications: the estimate
  across narrower and wider windows, and Rosenbaum bounds on the p-value.
- **Inference.** `knowledge/inference.yaml`, by facts: the point estimate from the conventional row, the interval from the
  robust bias-corrected row; clustered by the entity column when one exists, with the Bell-McCaffrey correction (cr2) when
  there are fewer clusters than `checks.yaml` trusts, which the `few_clusters` check also flags. Take-up that varies on one side
  only is passed to the library as `sharpbw`, so it selects the width as for a sharp design, and the `one_sided_takeup` check
  says so. No model call.
- **Evidence before the judgement.** The density rung (`checks.density_evidence`) runs rddensity once with the settings
  `checks.yaml` declares, splits the rows in nested windows either side of the line and tests each split as a coin toss, and
  writes the histogram; the line rung reads it from the ladder and its gate holds a clean verdict to it.
- **Balance before placement.** The balance rung (`checks.balance_evidence`) fits every numeric candidate's jump at the line and
  every category's share difference within the density's window before the covariates rung places them; the continuity check
  reads the rung for the columns the design keeps.
- **Validation.** `lane/checks.py` computes the pre-estimate facts: rows and effective rows a side, the density flag (read off the rung;
  the score's recorded orientation), mass points, support, compliance and the first stage, and continuity of every predetermined
  covariate at the cutoff. `knowledge/placebos.yaml` declares what runs after the estimate: placebo cutoffs on each side's own rows,
  the bandwidth grid, donuts, and two sensitivities with a range and no verdict, the polynomial order and the kernel at the
  design's window. Each reports its effective rows and counts as uninformative below the declared floor. A placebo entry applies by
  facts of the frozen design (the engine, the kind, the window rule), and every field it declares is read by code: a test greps
  for it, so no yaml line is a promise the lane does not keep.
- **The plot.** `rdplot`'s bins are written to `bins.csv`; `postviz.py` draws the outcome against the score from them.

## The rule every node follows

- **Facts**: `load`, `case`, `shape_table`, `merge_covariates`, `verify`, `threats`, `check_design`, `freeze_design`, `estimate`, `placebo`,
  `figures`, `assemble`. Declared inputs; a failed assumption is a typed stop.
- **Judgements**, bounded episodes or single calls, cited and gated: `score`, `line`, `covariates`, `heterogeneity` (episodes,
  each only for what the pack leaves open), `assess`, `pick_estimator`, `interpret`. A rung never asks the person beyond an
  incomplete rule; what it would not guess is a flag.
- **Gates that end a judgement on a fact**: a side the take-up shares contradict stops at `score`; a take-up jump that covers zero is
  a hard `no_first_stage` flag; `assess` cannot proceed without citing every flag, nor over a density or continuity flag without citing
  a note; `interpret` must state the estimand, bandwidth, effective rows and interval the design holds.
- **The pack is weighed first** by the harness's `case`; the person's beliefs become flags by `knowledge/beliefs.yaml`.
- **The model never touches the libraries.** `lane/adapter.py` is the only file that imports them.

## Files

```
family.yaml  design.py  handoff.py  probes.py  postviz.py
lane/        graph.py, nodes.py, prompts.py, contracts.py, state.py, adapter.py, checks.py, shape.py
lane/knowledge  estimators.yaml, inference.yaml, placebos.yaml, checks.yaml, beliefs.yaml
evals/  tests/
```

A run leaves `table.csv`, `canon.csv`, `scores.csv`, `design.json`, `design.md`, `bins.csv`, `artifacts.json`, `figures.json` and
`report.md` under `.artifacts/runs/<dataset>-rd-<id>/`.

## Run it

```bash
uv run python -m causal_agent.evals.lane discontinuity gov_transfers "Did receiving the transfer raise support for the government?"
```

```bash
uv run pytest causal_agent/families/discontinuity -q
```

## Library facts the code leans on (rdrobust 2.0.0, rddensity 3.0)

- `rdbwselect` takes no `level` argument; called with the same settings as the fit, its `mserd` row equals the fit's bandwidths.
- Passing `h` and `b` together keeps `b`; passing `h` alone resets `b = h`. The grid holds `b`.
- A cutoff not strictly inside the scores raises a plain `Exception`; fewer than 20 rows raises a `TypeError`; an empty side raises a
  `LinAlgError`. The adapter maps every exception to a typed fit error.
- "Mass points detected" is printed, not warned; the adapter captures stdout. Spurious matmul warnings on macOS are silenced and
  finiteness checked instead.
- `rddensity` fills only `t_jk` and `p_jk` under its default variance; its `p` underflows to 0 for large statistics; `repr` prints
  and warns, so it is never called. Its regularisation floor is 23 rows a side.
- A take-up column that is a step function of the side cannot be fitted as a first stage; the lane records it from the shares.
- `rdlocrand` 2.0 (local randomisation): `rdwinselect`, `rdrandinf` and `rdrbounds` return dicts; `rdwinselect` recommends no
  window without covariates; `rdrandinf` takes a fuzzy design as `[take_up, "ar"]` (a bare array breaks it) and its own `ci`
  option and `rdsensitivity` do not run in this port, so the adapter inverts the randomisation test over a grid itself; window
  lists must be numpy arrays; a window needs `wl < wr`.

## Known limits

- One cutoff. Many cutoffs (`rdmulti`), a kink design (`deriv=1`) and sampling weights each need a claim the interview does not ask.
- Rows with a missing outcome, score or take-up value are dropped at `shape_table` and counted.
