# causal_agent

One folder per step of the graph. Each step owns its code, its tests, and its evals.
Nothing in a step imports from a later step.

```
common/       what every step shares: the contracts (the context pack, the lane artifacts), the model wrapper, the address grammar
profile/      the deterministic profile of a CSV, the cards, the dataset index. No model.
memory/       what is known about a dataset: the map of fields, the catalogue, the gate, the checks, the probes, the family fit
desk/         the whole conversation: the question first, one question per turn to ready, the routing on the memory, the pack, the run, the chat after
viz/          figures as data with addresses: the spec, the pre-viz functions per family, the viz subgraph a Point goes to
knowledge/    what is available downstairs: the family registry, written as method knowledge, never as rules
router/       a shim over desk/route.py, plus the routing evals
specialists/  one subgraph per family, each turns a hand-off into a runnable analysis. dowhy/ (adjustment) and did/ (diff_in_diff, pyfixest) are built; the rest are stubs.
```

Order of the graph: desk (the journey, then the routing) → specialist, in its own process on the pack. The registry is read by the routing and lists the specialists.

End to end: `uv run python -m causal_agent.families.adjustment.lane.run students "Did completing the prep course raise math scores?"`.

Run all tests: `uv run pytest -q`. Run one step's tests: `uv run pytest causal_agent/desk -q`.
