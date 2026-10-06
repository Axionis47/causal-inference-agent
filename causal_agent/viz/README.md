# viz

Pictures: the figures a lane draws as data, and the sandbox a script runs in.

- `spec.py` — `FigureSpec`: a figure a lane draws from its own artifacts, as data with addresses (kind: bars, lines, points,
  density, interval, graph; series, marks, the note, what it draws on). The page draws it (`web/src/components/Figure.tsx`),
  the chat cites it (`figure:<id>.<series>.<i>`), a run keeps it in `figures.json`.
- `postviz/` — the post-run figures every lane can draw from a run record: the estimate against its falsifications, the effect
  by period. A family's own `postviz.py` draws more, inside its lane, with addresses the run checked.
- `sandbox.py` — `run(code, csv, out_dir)`: where the script runs is the config's choice. `VIZ_SANDBOX=auto` (the default) picks
  the strongest fence at hand: the macOS seatbelt (`sandbox-exec`: no network, reads limited to the interpreter, the system, the
  CSV and the artifact folder, writes to the artifact folder), `bwrap` or `unshare -rn` on Linux, else the same interpreter in
  isolated mode with a scrubbed environment. `VIZ_SANDBOX=docker`: the image from `make viz-image` (`docker/viz.Dockerfile`), no
  network, the CSV mounted read-only, the artifact folder as the working directory, a memory and CPU cap. Every kind has a
  timeout, hands back what the script printed, and sweeps everything but `code.py`, `figure.png`, `facts.json` and the small
  tables and notes it wrote; the outcome records the kind that ran. A kind named and not at hand refuses the run, never downgrades.
- What a script makes on request lives on the artifacts shelf (`memory/artifacts.py`); the exploration agent (`explore/`) is what
  writes the scripts and runs them here.

```bash
uv run pytest causal_agent/viz -q
```
