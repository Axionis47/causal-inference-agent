# causal_agent

One folder per step of the graph. Each step owns its code, its tests, and its evals.
Nothing in a step imports from a later step.

```
common/       what every step shares: the contracts (the context pack, the lane artifacts), the model wrapper, the address grammar
profile/      the deterministic profile of a CSV, the cards, the dataset index. No model.
memory/       what is known about a dataset: the field catalogue, the claim table, the checks, the probes, the family fit
intake/       the interview (moves into desk/ at stage 4) and shims for the modules that moved to profile/ and memory/
desk/         the context pack builder today; the whole conversation from CSV to run and back, stage by stage
knowledge/    what is available downstairs: the family registry, written as method knowledge, never as rules
router/       question + pack → which family, with reasons and citations. Ends at the hand-off.
specialists/  one subgraph per family, each turns a hand-off into a runnable analysis. dowhy/ (adjustment) and did/ (diff_in_diff, pyfixest) are built; the rest are stubs.
```

Order of the graph: intake → router → specialist. The registry is read by the router and lists the specialists.

End to end: `uv run python -m causal_agent.specialists.dowhy.run students "Did completing the prep course raise math scores?"`.

Run all tests: `uv run pytest -q`. Run one step's tests: `uv run pytest causal_agent/router -q`.
