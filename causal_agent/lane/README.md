# lane

The harness every lane runs on. What is shared is shape, not method: a lane keeps its engine, its judgements and its knowledge
yaml in its own package, and copies nothing that is here.

- `intake.py`: the table by code: the CSV the pack names, the columns the pack names, the scope's filter and window applied; what
  cannot be applied is a `Decline`.
- `case.py`: the pack weighed by code: facts, drafts, open, contested; the person's beliefs, unknowns and contradictions as flags by
  the lane's `beliefs.yaml`; `decide_by_code` for the stops and asks the yaml settles before any judgement.
- `verify.py`: a model's answer about a column against the pack's facts; every cite must resolve; a departure from the last
  reading names the claim and cites what changed it.
- `episode.py`: a judgement as a bounded episode: the model may look at the data through the tools, every fact it asked for gets
  an address (`probe:<node>.<n>`), the budget and the answer's shape are fixed by the caller, and the gate re-prompts with the
  log kept.
- `tools.py`: the read-only data tools an episode may be offered (describe, by arm, association, redundancy, cells, timing); no
  tool joins the outcome with the treatment before the design is frozen, by code.
- `ladder.py`: the shape every lane's ladder shares (one record per rung, each line addressed `ladder:<rung>.<field>`), the
  records every design shares (what a rung would not guess, a threat, heterogeneity), the threats every design carries from the
  pack, and the flags and declines the ladder yields.
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
