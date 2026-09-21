# causal_agent

One package per component. The layers, bottom to top, each importing only what is below it; the rule is held by
`import-linter` (see `docs/architecture.md`).

```
common/       what every step shares: the contracts (the pack, the artifacts, the frame), the model wrapper, the address grammar, the config
profile/      the deterministic profile of a CSV, the cards, the dataset index. No model.
memory/       what is known about a dataset: the map of fields, the catalogue, the gate, the checks, the fit grid, the text views
viz/          figures as data with addresses: the spec, the viz subgraph a Point goes to, the post-run figures every lane can draw
lane/         the harness every lane is built on: intake, case, verify, asks, records, figures, words, the shared nodes and loaders
families/     one package per family: knowledge, design block, probes, figures, the lane, evals, tests; declared/ for the ones with no lane; registry.py
desk/         the whole conversation: the question first, one question per turn to ready, the routing on the memory, the pack, the run, the chat after
server/       the FastAPI shell: datasets, sessions, the transcript, the projection for the page
evals/        the eval runner and the command-line runs, outside the layers
```

Order of the graph: desk (the journey, then the routing) → a family's lane, in its own process on the pack. The desk
reaches a family only through `families.registry`.

End to end: `uv run python -m causal_agent.evals.lane adjustment students "Did completing the prep course raise math scores?"`.

Run all tests: `make test`. Run one package's tests: `uv run pytest causal_agent/desk -q`. Everything CI runs: `make check`.
