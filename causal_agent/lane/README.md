# Shared analysis runtime

The harness shared by the specialized analysis workflows. Each family retains its causal endpoint declarations, judgments, engine adapters and frozen execution policy.

Start with the [component chapter](../../docs/components/10-analysis-runtime.md) for responsibility boundaries, contracts, storage and verification. The current modules are:

| Modules | Responsibility |
|---|---|
| `graph.py`, `rungs.py`, `state.py`, `ladder.py` | Build named graphs from declared rungs and supply shared state/record shapes |
| `intake.py`, `endpoints.py` | Read the admitted table/scope and validate context, dataset and catalogue identities |
| `episode.py`, `tools.py` | Bounded model episodes, permitted data tools, addressed evidence and recorded gate attempts |
| `ledger.py` | Source-classified evidence and accepted revision history |
| `knowledge.py`, `space.py` | Load capability declarations and evaluate declared support limits |
| `nodes.py`, `terminal.py` | Shared node utilities, explicit stops and result status |
| `figures.py`, `records.py` | Validate figure sources and assemble common result/artifact records |
| `prompts.py`, `words.py` | Shared prompt rules and question-oriented check wording |

The desk launches the selected family graph in a separate process through `desk.pipeline`. The family supplies its nodes to this graph builder. Essential missing context returns a `ContextDeficiency` to the desk; the lane does not interview the person.

The [generated ladders](../../docs/ladders) are the detailed trajectory reference. [Gates](../../docs/gates.md) explains model and code authority. Family diagnostic code reuses `explore/diagnostics.py`; it is supplied by family callers rather than imported upward from the harness.

```bash
LANGSMITH_TRACING=false uv run pytest causal_agent/lane -q
```
