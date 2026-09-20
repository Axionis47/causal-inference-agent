# specialists/rd — the discontinuity lane

Entered from the router's hand-off for the `discontinuity` family. Turns a question, a pack, and a table into a canonical cutoff table, a frozen Design, an rdrobust fit, falsifications, and a report, or an honest Feasibility stop.

```
load ─ score ─ shape_table ─(relate × N)─ merge_covariates ─ verify ─ check_design
     ─(assess, only when flagged)─ pick_estimator ─ freeze_design ─ estimate ─(placebo × K)─ interpret ─ assemble
any typed stop ───────────────────────────────────────────────────────────▶ feasibility ─ assemble
```

N = relevant columns other than the score, the take-up column, the outcome, and the entity column. K = placebos the catalogue declares. Neither appears in the graph.

## How it sits on rdrobust and rddensity

The pair has five independent axes and the lane maps one artifact onto each.

- **Canonical inputs.** `shape.py` produces `y, x, side` plus `t` (take-up) when a column records receipt, `cluster` when the dataset declares an entity, and every numeric candidate covariate. `x` is the score recentred on the cutoff and flipped so the treated side is positive, so the cutoff is 0 everywhere below and every fit, check, and placebo is written for one geometry. When the notes say a unit exactly at the cutoff is control (a strict rule), the effective cutoff moves to the midpoint between the cutoff and the next distinct score on the treated side, and the shift is a recorded fact; the library would otherwise count those rows as treated.
- **Local polynomial spec.** `knowledge/estimators.yaml` holds named specs over the canonical names with the *Foundations*' defaults pinned: local linear, triangular kernel, MSE-optimal bandwidth, mass points adjusted. Each entry carries its estimand label. The bandwidth is the library's choice, except when the score has fewer distinct values than `checks.yaml` declares: then `h` is the distance that keeps three support points on each side and the Design says so.
- **Inference.** `knowledge/inference.yaml` picks the point estimate from the Conventional row and the interval from the Robust (bias-corrected) row, and clusters by the pack's entity column when one exists. No model call.
- **Validation.** `checks.py` computes the pre-estimate facts: rows and effective rows a side, the density test (rddensity, on every row with a finite score, in the score's recorded orientation, because the library's statistic is not symmetric under a sign flip when scores tie), mass points, support, compliance and the first stage, and continuity of every predetermined covariate at the cutoff. `knowledge/placebos.yaml` declares the falsifications that run after the estimate: placebo cutoffs on each side's own rows with the sharp reduced form, the bandwidth grid (h_CER, h_MSE, and twice each, with the bias bandwidth held), and donuts. Every one reports its effective rows and counts as uninformative below the declared floor; the pass rules cannot fail a true null.
- **The plot.** `rdplot`'s bins are written to `bins.csv` for the visualisation step. Nothing draws.

## The rule every node follows

- **Facts**: `load`, `shape_table`, `merge_covariates`, `check_design`, `freeze_design`, `estimate`, `placebo`, `assemble`. Declared inputs; a failed assumption is a typed stop.
- **Judgements**, one model call each, cited and gated: `score`, `relate`, `assess`, `pick_estimator`, `interpret`. Two decide what runs: `score` and the estimator pick.
- **Gates that end a judgement on a fact**: a side the take-up shares contradict stops at `score`; a take-up column whose jump at the cutoff covers zero is a hard `no_first_stage` flag; `assess` cannot proceed without citing every flag, and cannot proceed over a density or continuity flag without citing a note; `interpret` must state the estimand, bandwidth, effective rows, and interval the Design and estimate hold, and cite every soft flag and every failed falsification.
- **Design freeze** before any estimate; no loop after an estimate exists.
- **The model never touches the libraries.** `adapter.py` is the only file that imports them.

## Files

`contracts.py`, `state.py`, `prompts.py`, `nodes.py`, `graph.py`, `adapter.py`, `checks.py`, `shape.py`, `run.py`, `knowledge/` (estimators, inference, placebos, checks and their loader), `tests/`, `evals/` (LangSmith dataset `causal-rd-v0`).

Runs write to `RUN_DIR/<dataset>-rd-<id>/`: `table.csv` (the whole file), `canon.csv`, `scores.csv`, `design.json`, `design.md`, `bins.csv`, `artifacts.json`, `report.md`.

## Run it

```bash
uv run python -m causal_agent.families.discontinuity.lane.run gov_transfers "Did receiving the transfer raise support for the government?"
uv run python -m causal_agent.families.discontinuity.lane.run --handoff causal_agent/families/discontinuity/evals/handoffs/students_forced.json
uv run pytest causal_agent/families/discontinuity/lane -q
```

## Library facts the code leans on (rdrobust 2.0.0, rddensity 3.0)

- `rdbwselect` takes no `level` argument; called with the same `fuzzy`, `cluster`, `covs`, kernel, order, and mass-point setting as the fit, its `mserd` row equals the fit's bandwidths exactly.
- Passing `h` and `b` together keeps `b`; passing `h` alone resets `b = h`. The grid holds `b`.
- A cutoff not strictly inside the passed scores raises a plain `Exception`; fewer than 20 rows in total raises a `TypeError`; a bandwidth that holds no rows on a side raises a `LinAlgError`. The adapter maps every exception to a typed fit error.
- "Mass points detected" is printed, not warned, when a fifth or more of a side's scores are duplicates; the adapter captures stdout. Every call on macOS emits spurious matmul warnings; the adapter silences them and checks finiteness instead.
- `rddensity` fills only `t_jk` and `p_jk` under its default variance; its `p` underflows to 0 for large statistics; `repr` prints and warns, so it is never called. Its regularisation floor is 23 rows a side, below which the bandwidth covers a whole side and the test is not local.
- A take-up column that is a step function of the side cannot be fitted as a first stage; the lane records the first stage from the shares instead.

## Known limits of this version

- Local randomisation for scores with very few support points is declared, not built (no verified Python port of rdlocrand). Such scores run under the support-points bandwidth rule, with the `support` flag carried.
- One cutoff. Multiple cutoffs and kink designs are not handled.
- Clean data assumed: rows with a missing outcome, score, or take-up value are dropped at `shape_table` and counted; the density test still sees every row with a score.
