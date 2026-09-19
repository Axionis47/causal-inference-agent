# viz

Figures as data with addresses. A figure is never an image: the page draws it (`web/src/components/Figure.tsx`), the chat
cites it (`figure:<id>.<series>.<i>`), a run keeps it. See docs/desk-redesign.md, section 4.

- `spec.py` — `FigureSpec` (kind: bars, lines, points, density, interval; series, marks, the note, what it draws on), `Point`
  (what the desk wants shown: a claim in the question's words, never a figure name), `Figure` (made with the probe number
  from the same computation, or refused with why).
- `previz/` — one module per family, pure over a table and named columns: `adjustment.overlap` (shares by arm at every
  level the offer depended on, and the overlap probe from the same cells), `diff_in_diff.by_group_over_time` (the outcome
  per period by group, the change marked, the pre-periods counted), `discontinuity.density` and `outcome_by_bin` (rows and
  the outcome per score bin either side of the cutoff, the rows a side counted).
- `figures.yaml` — what each function shows, when it makes a point, and what it needs settled in the memory. Declarations;
  no dataset name, no rule.
- `graph.py` — the viz subgraph: `candidates` (code: the declared figures whose family matches and whose needs the memory
  meets), `pick` (a judgement only among several; one is chosen by code, none is a refusal), `render` (the function on the
  table), `check` (code: values exist, every address resolves). `make(point, dataset, outcome)` runs it as one call.

```bash
uv run pytest causal_agent/viz -q
```
