# evals

The eval runner and the command-line runs, outside the layers. The only place a model runs on purpose.

- `spec.py`: `EvalSpec`, what the runner needs to know about a family's evals; each family declares one under `families/<name>/evals/`.
- `families.py`: the families that have evals, an explicit list like the registry.
- `lane.py`: one run of a family's lane from the command line, routed first or from a stored hand-off.
- `run.py`: a family's cases against its LangSmith dataset, scored. `make evals FAMILY=adjustment`.
- `dataset.py`: upload a family's `cases.yaml` to its LangSmith dataset.
- `evaluators.py`: the evaluators every family shares.

```bash
uv run python -m causal_agent.evals.lane adjustment students "Did completing the prep course raise math scores?"
```

```bash
uv run pytest causal_agent/evals -q
```

See [docs/testing.md](../../docs/testing.md).
