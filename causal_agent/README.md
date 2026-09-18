# causal_agent

One folder per step of the graph. Each step owns its code, its tests, and its evals.
Nothing in a step imports from a later step.

```
common/       what every step shares: the contracts (typed artifacts every node reads and writes) and the model wrapper
intake/       CSV + semantic note → a citable pack. Profiler, pack loader, dataset index. Deterministic, no model.
knowledge/    what is available downstairs: the family registry, written as method knowledge, never as rules
router/       question + pack → which family, with reasons and citations. Ends at the hand-off.
specialists/  one subgraph per family, each turns a hand-off into a runnable analysis. dowhy/ (adjustment) and did/ (diff_in_diff, pyfixest) are built; the rest are stubs.
```

Order of the graph: intake → router → specialist. The registry is read by the router and lists the specialists.

End to end: `uv run python -m causal_agent.specialists.dowhy.run students "Did completing the prep course raise math scores?"`.

Run all tests: `uv run pytest -q`. Run one step's tests: `uv run pytest causal_agent/router -q`.
