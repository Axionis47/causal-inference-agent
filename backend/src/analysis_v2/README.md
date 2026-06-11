# analysis_v2 (rebuild target)

The analysis tail, rebuilt from scratch against the finalized start-analysis
payload (`docs/analysis-slice/00-start-analysis-payload.md`). Kept in its own
folder, separate from the input slice (download + data-review gate).

Planned structure:
  state/    the in-memory STATE the analysis reads/seals (the payload + outputs)
  agents/   one folder per analysis agent (EDA, causal-structure, estimation, ...)
  runner/   the seam that loads a CONFIRMED record and drives the agents

Nothing here yet. Built one agent at a time, each with its own design doc.
