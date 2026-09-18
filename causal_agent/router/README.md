# router

Question + pack → the family of analysis, with reasons that cite the pack. The graph ends at the hand-off.

```
load_pack ─ [prefilter × N, wide tables only] ─ frame ─ [test_family × F] ─ decide ─ gate ─ handoff ─ specialist_<family>
```

- `state.py` — RouterState, worker task schemas, runtime Context.
- `nodes.py` — the nodes. Deterministic ones (load_pack, gate, handoff) never call a model. Model ones (prefilter, frame, test_family, decide) call `structured()` once.
- `prompts.py` — no column names, no family names, no rules. Cards and knowledge go in as context.
- `graph.py` — the StateGraph. `graph` for `langgraph dev`, `compile_local()` for tests and scripts.
- `run.py` — CLI: `uv run python -m causal_agent.router.run <dataset> "<question>" [--json]`. Prints the decision record.
- `tests/` — the graph with a fake model: routing, the gate's citation loop, the no-family exit.
- `evals/` — five real cases, one per lane (`cases.yaml`); LangSmith dataset upload (`dataset.py`), code evaluators (`evaluators.py`), and the runner (`run.py`). Needs `LANGSMITH_API_KEY`.

The router never assigns causal roles to columns; that is lane vocabulary and belongs to the specialist.
Thoughts from the model land in `state.debug` and the record's debug section. Never gated, cited, or read by another node.
