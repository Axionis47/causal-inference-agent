# specialists/did — the diff-in-diff lane

Entered from the router's hand-off for the `diff_in_diff` family. Turns a question, a pack slice, and a table into a canonical panel, a frozen Design, a pyfixest fit, placebo tests, and a report, or an honest Feasibility stop.

```
load ─ groups ─ periods ─ shape_table ─(relate × N)─ merge_controls ─ verify ─ check_design
     ─(assess, only when flagged)─ pick_estimator ─ freeze_design ─ estimate ─(placebo × K)─ interpret ─ assemble
any typed stop ───────────────────────────────────────────────────────────▶ feasibility ─ assemble
```

N = relevant columns other than the group, time, and outcome columns. K = placebos the catalogue declares. Neither appears in the graph.

## How it sits on pyfixest

pyfixest has four independent axes and the lane maps one artifact onto each.

- **The canonical panel.** `shape.py` produces `y, unit, time, treated, post, treat, rel_time, cohort` from either a long table (a time column) or a wide one (a before and an after column, one synthetic unit per row). Every DID helper in the library reads this shape.
- **Formula shapes.** `knowledge/estimators.yaml` holds formulas over the canonical names. Static: `y ~ treat | unit+time`. Dynamic: `y ~ i(rel_time, treated, ref=-1) | unit+time`. Controls are filled in by the adapter; the static shape uses `csw0()` so the report shows the effect with no controls and with each control added, which is the library's own specification curve.
- **Inference.** `knowledge/inference.yaml` picks the `vcov` and any resampling from facts: robust errors on wide two-period data, cluster by unit with many treated units, a wild cluster bootstrap with few, and the placebo-group p-value as the inference when there is one treated unit.
- **Adoption pattern.** `cohorts` is a fact from the panel. One cohort is the built path. The staggered entries (`did2s`, `lpdid`) are declared in the catalogue and gated by `cohorts_min`; `staggered` is a hard check until they are exercised on a staggered dataset.

Diagnostics come from the library where it has them: the pre-trends check is a joint Wald test on the lead coefficients of the dynamic fit. Placebos refit the bare formula on a perturbed panel: the treated label reassigned across units (`placebo_group`), or a fake change in the middle of the pre-window (`placebo_timing`). The library's fast randomisation path is not used: it permutes rows, not units, and its compiled kernel fails on two-period panels.

## The rule every node follows

- **Facts**: `load`, `shape_table`, `merge_controls`, `check_design`, `freeze_design`, `estimate`, `placebo`, `assemble`. Declared inputs; a failed assumption is a typed stop.
- **Judgements**, one model call each, cited and gated: `groups`, `periods`, `relate`, `assess`, `pick_estimator`, `interpret`. Two decide what runs: `periods` and the estimator pick.
- **Design freeze** before any estimate; loops only on failure facts.
- **The model never touches pyfixest.** `adapter.py` is the only file that imports it.

## Files

`contracts.py`, `state.py`, `prompts.py`, `nodes.py`, `graph.py`, `adapter.py`, `checks.py`, `shape.py`, `run.py`, `knowledge/` (estimators, inference, placebos, checks and their loader), `tests/`, `evals/` (LangSmith dataset `causal-did-v0`).

Runs write to `RUN_DIR/<dataset>-did-<id>/`: `table.csv`, `panel.csv`, `design.json`, `design.md`, `artifacts.json`, `report.md`.

## Run it

```bash
uv run python -m causal_agent.families.diff_in_diff.lane.run card_krueger "Did New Jersey's 1992 minimum wage rise reduce fast food employment?"
uv run python -m causal_agent.families.diff_in_diff.lane.run --handoff causal_agent/families/diff_in_diff/evals/handoffs/cigar_forced.json
uv run pytest causal_agent/families/diff_in_diff/lane -q
```

## Known limits of this version

- One cohort only. Staggered adoption stops at `check_design` with a hard flag; the catalogue entries exist but are not exercised on our data.
- Fixed effects are spelled `unit+time` without spaces on purpose: pyfixest 0.60 splits the fixed-effects string on `+` without stripping.
- The dynamic model's summary number is the mean of the post-period coefficients; the per-period values are in the report and `artifacts.json`.
- Clean data assumed: rows with a null in any relevant column are dropped at `load`.
