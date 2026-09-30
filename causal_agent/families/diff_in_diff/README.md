# families/diff_in_diff: the pyfixest lane

Effect of a change that reached some units at a date, compared with units it did not reach, before and after. The desk hands the
lane a pack; the lane builds a canonical panel, freezes a design, fits it on pyfixest, runs the placebos, and writes a report or an
honest stop.

```
load ─ case ─ groups ─ periods ─ shape_table ─ comparison ─ controls ─ merge_controls ─ verify ─ heterogeneity ─ threats
     ─ check_design ─(assess, only when flagged)─ pick_estimator ─ freeze_design ─ estimate ─(placebo × K)─ interpret
     ─ figures ─ assemble
any typed stop, or a question back to the desk ─────────────────────────────▶ feasibility ─ figures ─ assemble
```

K is the placebos the catalogue declares. `groups`, `periods`, `shape_table`, `comparison`, `controls`, `heterogeneity` and
`threats` are the rungs of the ladder (`lane/contracts.py`, `Ladder`), and `freeze_design` adds the clustering rung by code: who
got the change, the clock, the shape of the panel, whether the comparison group is a fair stand-in and what could break it, the
candidate controls placed together, where the effect could differ, the risks, where the errors cluster. A rung is code where the
pack settles it and a bounded episode where it does not, with the data tools and the budgets `knowledge/checks.yaml` declares.

## How it sits on pyfixest

pyfixest has four independent axes and the lane maps one artifact onto each.

- **The canonical panel.** `lane/shape.py` produces `y, unit, time, time_index, treated, post, treat, rel_time, cohort` from a long
  table (a time column) or a wide one (a before and an after column, one synthetic unit per row). `cohort` is each unit's first
  treated period as a 1-based index, 0 for never treated, which is what pyfixest's DID estimators call `gname`; it is read from the
  pack's adoption column when it names one, else from a treatment indicator that switches on within a unit and stays on, else from
  the treated label and the one change period. One first period is one-shot adoption; several is staggered, and the shape facts say
  which units got the change when and whether any unit is never treated. Every helper reads this shape.
- **Formula shapes.** `knowledge/estimators.yaml` holds formulas over the canonical names. Static: `y ~ treat | unit+time`. Dynamic:
  `y ~ i(rel_time, treated, ref=-1) | unit+time`. Controls are filled in by the adapter; the static shape uses `csw0()` so the report
  shows the effect with no controls and with each control added.
- **Inference.** `knowledge/inference.yaml` picks the `vcov` and any resampling from facts: robust errors on wide two-period data,
  cluster by unit with many treated units, a wild cluster bootstrap with few, the placebo-group p-value when there is one treated unit.
- **Adoption pattern.** `cohorts` is a fact from the panel. One cohort is the built path; the staggered entries are declared and gated
  by `cohorts_min`, and `staggered` is a hard check until they are exercised on a staggered dataset.

The pre-trends check is a joint Wald test on the lead coefficients of the dynamic fit. Placebos refit the bare formula on a perturbed
panel: the treated label reassigned across units, or a fake change in the middle of the pre-window.

## The rule every node follows

- **Facts**: `load`, `case`, `shape_table`, `merge_controls`, `verify`, `threats`, `check_design`, `freeze_design`, `estimate`,
  `placebo`, `figures`, `assemble`. Declared inputs; a failed assumption is a typed stop.
- **Judgements**, bounded episodes or single calls, cited and gated: `groups`, `periods`, `comparison`, `controls`,
  `heterogeneity` (episodes, each only for what the pack leaves open), `assess`, `pick_estimator`, `interpret`. A rung never asks
  the person beyond the first period; what it would not guess is a flag.
- **The pack is weighed first** by the harness's `case`; the person's beliefs become flags by `knowledge/beliefs.yaml`.
- **Design freeze** before any estimate; loops only on failure facts.
- **The model never touches pyfixest.** `lane/adapter.py` is the only file that imports it.

## Files

```
family.yaml  design.py  handoff.py  probes.py  postviz.py
lane/        graph.py, nodes.py, prompts.py, contracts.py, state.py, adapter.py, checks.py, shape.py
lane/knowledge  estimators.yaml, inference.yaml, placebos.yaml, checks.yaml, beliefs.yaml
evals/  tests/
```

A run leaves `table.csv`, `panel.csv`, `design.json`, `design.md`, `artifacts.json`, `figures.json` and `report.md` under
`.artifacts/runs/<dataset>-did-<id>/`.

## Run it

```bash
uv run python -m causal_agent.evals.lane diff_in_diff card_krueger "Did New Jersey's 1992 minimum wage rise reduce fast food employment?"
```

```bash
uv run pytest causal_agent/families/diff_in_diff -q
```

## Known limits

- One cohort only. Staggered adoption stops at `check_design` with a hard flag.
- Fixed effects are spelled `unit+time` without spaces on purpose: pyfixest 0.60 splits the string on `+` without stripping.
- The dynamic model's summary number is the mean of the post-period coefficients; the per-period values are in the report.
- Rows with a null in any relevant column are dropped at `load`.
