# profile

The deterministic profile of a CSV and the cards built from it. No model.

- `profiler.py`: one CSV in, one `Profile` out: rows, columns, duplicate rows, the grain the keys suggest, time coverage, and per
  column its kind, nulls, distinct values, bounds, top values, hints.
- `data.py`: the file and its profile, loaded once per path in this process and cached once per file content on disk.
- `pack.py`: a semantic note and a profile joined into citable cards, one line per fact with its `col:<key>.profile.<facet>` address.
- `datasets.py`: the dataset index (`data/datasets.yaml`) and the csv path rule.

```bash
uv run pytest causal_agent/profile -q
```

## Owned dataset inspection (Phase 3)

`inspection.py` reads original CSV fields as text and emits the contracts in `report.py`. It reports ambiguity/failure explicitly and never writes a converted dataset. The server persists dataset and column results separately in PostgreSQL. This entrypoint is independent of the older analysis `Profile`/cache contract. Shared sentinel constants remain in `profiler.py`; no new profiling library or model is used.
