# viz

Pictures, two kinds.

- `spec.py` — `FigureSpec`: a figure a lane draws from its own artifacts, as data with addresses (kind: bars, lines, points,
  density, interval, graph; series, marks, the note, what it draws on). The page draws it (`web/src/components/Figure.tsx`),
  the chat cites it (`figure:<id>.<series>.<i>`), a run keeps it in `figures.json`.
- `postviz/` — the post-run figures every lane can draw from a run record: the estimate against its falsifications, the effect
  by period. A family's own `postviz.py` draws more, inside its lane, with addresses the run checked.
- `draw.py` — the drawing tool. `draw(DrawRequest) -> (Artifact | None, Decline | None, thoughts)`: the model writes one Python
  script that draws one figure from the CSV and writes every number it shows to `facts.json`; the script runs in the sandbox;
  three tries, a failed try's error in the next prompt; three failures are a `Decline` and no folder. The prompt names no method
  and no column; the columns and what is known arrive as data.
- `sandbox.py` — `run(code, csv, out_dir)`: where the script runs is the config's choice. `VIZ_SANDBOX=auto` (the default) picks
  the strongest fence at hand: the macOS seatbelt (`sandbox-exec`: no network, reads limited to the interpreter, the system, the
  CSV and the artifact folder, writes to the artifact folder), `bwrap` or `unshare -rn` on Linux, else the same interpreter in
  isolated mode with a scrubbed environment. `VIZ_SANDBOX=docker`: the image from `make viz-image` (`docker/viz.Dockerfile`), no
  network, the CSV mounted read-only, the artifact folder as the working directory, a memory and CPU cap. Every kind has a
  timeout and sweeps everything but `code.py`, `figure.png`, `facts.json`; the outcome and the artifact record the kind that ran.
  A kind named and not at hand refuses the drawing, never downgrades.
- `store.py` — where a drawn `Artifact` lives and how it is addressed: `data/memory/<name>/viz/pre/<id>/` before any run,
  `data/memory/<name>/designs/<n>/viz/<id>/` after run n, so it is deleted with the dataset or the design. The chat cites
  `artifact:<id>` and `artifact:<id>.<fact>`; code never reads a picture back.

```bash
uv run pytest causal_agent/viz -q
```
