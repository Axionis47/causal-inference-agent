# explore

The exploration agent: a read-only look at the data on request, from any turn of the conversation.

- `contracts.py` — `ExploreRequest` (the table, what is known, the columns, the pair it must not join), `Exploration` (the
  record: the step whose files are the answer, the facts, the picture, the table), `Step`, `Trail`.
- `tools.py` — the one tool, `run_code`: a script runs in the sandbox (`viz/sandbox.py`) in its own `step_<n>/` folder; what it
  printed, the files it wrote and its `facts.json` come back as the tool's text, addressed `probe:explore.<n>` by the episode.
- `gates.py` — what code checks: the script parses and does not name the held-back column; the record names a step that ran,
  facts that are in its `facts.json`, a picture that exists, a table that parses, at least one product, nothing that relates the
  outcome to the treatment before the freeze, and no number the facts or the context do not hold.
- `project.py` — the no-join rule for free-form code: before the freeze the table sent has the outcome or the treatment taken
  out, by code; an ask naming both is refused before any model call.
- `agent.py` — one exploration in this process: `explore(req) -> Trail`, over `lane/episode.py::run_episode` with the budget,
  the tries and the timeout `checks.yaml` declares.
- `prompts.py` — the one judgement's text. Method-free and column-free.

The desk reaches it through its client (`client.py`), which allocates the artifact's folder, projects the table, sends the
request to the agent's process over A2A, and writes what came back to the artifacts shelf (`memory/artifacts.py`).
