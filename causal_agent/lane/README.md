# lane

The harness every lane runs on. What is shared is shape, not method: a lane keeps its engine, its judgements and its knowledge
yaml in its own package, and copies nothing that is here.

- `intake.py`: the table by code: the CSV the pack names, the columns the pack names, the scope's filter and window applied; what
  cannot be applied is a `Decline`.
- `case.py`: the pack weighed by code: facts, drafts, open, contested; the person's beliefs, unknowns and contradictions as flags by
  the lane's `beliefs.yaml`; `decide_by_code` for the stops and asks the yaml settles before any judgement.
- `verify.py`: a model's answer about a column against the pack's facts; every cite must resolve.
- `asks.py`: one question back to the desk, the same shape in every lane.
- `figures.py`: the figures a lane leaves, each checked against the addresses this run can cite.
- `records.py`: `artifacts.json` with the common keys, the result the desk reads, the report's tail.
- `nodes.py`: the stream writer, the honest stop, the cards, the frame text every judgement reads first, the settled and drafted
  blocks a relate prompt shows, the shared relate node.
- `prompts.py`: the pick and interpret prompts, the cite rule, the plain-words rule.
- `knowledge.py`: a lane's yaml files, read once and rendered for the model.
- `graph.py`: how a lane's graph is compiled, as a node inside the desk or standalone.
- `state.py`: the state keys every lane shares and the reducers that keep a fan-out honest.
- `words.py`: what a check asks, in the question's words, from `in_words` in the lane's `checks.yaml`.

Read [docs/lanes.md](../../docs/lanes.md) for the harness in use and [docs/gates.md](../../docs/gates.md) for the gates.

```bash
uv run pytest causal_agent/lane -q
```
