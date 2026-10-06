# viz

Pictures: the figures a lane draws as data.

- `spec.py` — `FigureSpec`: a figure a lane draws from its own artifacts, as data with addresses (kind: bars, lines, points,
  density, interval, graph; series, marks, the note, what it draws on). The page draws it (`web/src/components/Figure.tsx`),
  the chat cites it (`figure:<id>.<series>.<i>`), a run keeps it in `figures.json`.
- `postviz/` — the post-run figures every lane can draw from a run record: the estimate against its falsifications, the effect
  by period. A family's own `postviz.py` draws more, inside its lane, with addresses the run checked.
- What a script makes on request lives on the artifacts shelf (`memory/artifacts.py`); the exploration agent (`explore/`) writes
  the scripts and runs them in its sandbox (`explore/sandbox.py`).

```bash
uv run pytest causal_agent/viz -q
```
