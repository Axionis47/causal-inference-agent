# memory

What is known about one dataset, with a status and a source on every field. Built through the conversation, checked
against the file, projected into every run's context pack. See docs/desk-redesign.md, section 1.

- `records.py` — the map. `Field` (value, status, source, the person's words, the checks that touched it); `Column` (the
  profiler's facts on one column, never written to); `Memory` = `{address: Field}` beside the columns, the dataset facts, and
  the person's words. An address absent from the map has not been spoken of. A memory projects to the older claim table and
  reads one in (`to_claims`, `from_claims`) so the checks, the probes, the fit table, and the interview run on it unchanged
  until they are rewritten on the map.
- `ops.py` — the facts: `seed` from the profile; `apply`, the one gated write path (a source is required, a model may only
  draft, a confirmed field changes only on the person's word or a data check, a belief only on the person's word, values
  must fit the field); `roles`, a view of what each column is to the question, computed every time and never written;
  `check` (the data checks plus the consistency rules: a column the offer looked at is set before it, a score is before,
  the outcome is after, a before-column cannot be moved by the change; a failed rule marks the field refuted and never
  overwrites the value); `probe`; `fit`; `open` (what is still vague, the question engine's input).
- `store.py` — `data/memory/<name>/` (meta, columns, fields, said, designs/); `migrate` from the older claims files.
- `fields.yaml`, `checks.yaml`, `catalogue.py` — the field catalogue: kinds, fields, options, frames, checks, family needs, thresholds.
- `claims.py`, `table.py`, `checks.py`, `probes.py` — the claim-table view and the code that still runs on it.

```bash
uv run python -m causal_agent.memory.store migrate --all
uv run python -m causal_agent.memory.store show students3
uv run pytest causal_agent/memory -q
```
