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
