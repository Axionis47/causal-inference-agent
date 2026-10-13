# The exploration agent

A look at the data on request, from any turn of the conversation, before or after the run. The model that reads each message
decides that the person asked to look at something; a separate agent, in its own process, writes scripts, runs them in a
sandbox one step at a time, and answers one record; code gates the record; what it made lands on the artifacts shelf with an
address the chat can cite. Before freeze it describes permitted non-outcome data. After a matching frozen design, it can discuss and visualize the run and its data.

## 1. How a request is detected

Two judgements can raise it, neither of them the agent's.

**Before the run**, the Reader reads every message into a `Reading` ([desk/contracts.py](../causal_agent/desk/contracts.py)) and
says its kind first: update, explore, ask or run. An `explore` reading carries what to look at in the person's words; so may an
`update` that asks for a picture in the same breath. The `infer` node keeps the request across retries and routes to the
`explore` node ([desk/nodes/interview.py](../causal_agent/desk/nodes/interview.py)). Nothing is asked after it: the reply is the
summary with the artifact beside it.

**After the run**, the Explainer's reply is an `AfterReply` whose `kind` is one of answer, revise, what_if, requestion, explore,
done. An `explore` reply with an empty `explore` field is refused by the gate ([desk/explainer.py](../causal_agent/desk/explainer.py)).
The `turn` node maps it to `explore_after` ([desk/nodes/after.py](../causal_agent/desk/nodes/after.py)). The Explainer can also
raise it before the run, since `answer` and `explore` are the only two kinds legal there.

## 2. What the agent is told, and what it is not

`explore_context` ([desk/nodes/interview.py](../causal_agent/desk/nodes/interview.py)) builds the context from the memory's text
view: the dataset, the change and the assignment, the beliefs, then one line per non-constant column. After a run, `explore_after`
swaps the dataset part for the pack's own rendering and appends the run's material lines that start with `design:`, `estimate:`,
`check:`, `falsification:` or `sensitivity:`, so a number the agent quotes about the run is one the run holds.

**The pre-freeze outcome fence.** Unrestricted scripts could reconstruct treatment from a score, instrument, group or adoption date. Before a matching design freezes, the table sent to the agent always has the outcome removed by code ([explore/project.py](../causal_agent/explore/project.py)). Outcome-only requests cannot bypass this fence. Requests explicitly joining treatment and outcome are refused before a model call. Deterministic outcome summaries remain available from the profile.

The withheld outcome is removed from the column inventory and rejoined by row identity after an allowed table transformation. A failed run without frozen designs does not lift the fence. After freeze, the desk verifies that the design belongs to the current dataset bytes and version before exposing the full table.

## 3. The loop

One tool, `run_code` ([explore/tools.py](../causal_agent/explore/tools.py)): a complete script runs in the sandbox in a fresh
`step_<n>/` folder under the artifact's; what it printed, cut to its tail, the files it wrote and the numbers in its `facts.json`
come back as the tool's text, addressed `probe:explore.<n>`. The loop is the lane's bounded episode
([lane/episode.py](../causal_agent/lane/episode.py)), with one difference: the agent is looked at one step at a time as a
structured record (`RunCodeStep`: the next script, or done), not as a function call, because Gemini answers
`MALFORMED_FUNCTION_CALL` when a function call carries a multiline script. The budget, the tries and the timeout are declared
in [explore/checks.yaml](../causal_agent/explore/checks.yaml).

The system prompt is in [explore/prompts.py](../causal_agent/explore/prompts.py). It names no method and no column; the ask, the
context and the columns arrive as data.

## 4. The record and the gate

| step | what | where |
|---|---|---|
| in | `ExploreRequest`: the dataset, the ask, the context, the columns sent, the table's path, the folder, the pair, which was held back, the freeze, the design, the versions | [explore/contracts.py](../causal_agent/explore/contracts.py) |
| each step | `RunCodeStep`: the next script, or done; gate: the script parses and does not name the held-back column | [explore/gates.py](../causal_agent/explore/gates.py) |
| judgement | `Exploration`: the summary, the step whose files are the answer, the facts stood behind, whether its figure is the picture, the table file | [explore/contracts.py](../causal_agent/explore/contracts.py) |
| gate | the step ran; its `facts.json` is an object of numbers holding every promised name; the picture exists when claimed; the table parses; at least one product; before the freeze no sentence and no fact name relates the outcome to the treatment; every number in the summary is within one percent of a fact or of a number in the context | [explore/gates.py](../causal_agent/explore/gates.py) |
| out | an `Artifact` on the shelf, or after three refusals a `Decline(check="explore.failed")` and no folder | [memory/artifacts.py](../causal_agent/memory/artifacts.py), [explore/client.py](../causal_agent/explore/client.py) |

## 5. The process and the wire

The agent is an A2A server ([explore/server.py](../causal_agent/explore/server.py)): a card with one skill at
`/.well-known/agent-card.json` and a JSON-RPC endpoint at the root. The desk's client ([explore/client.py](../causal_agent/explore/client.py))
starts one process per call under the desk's run and thread (`trace.child_env`; the server opens under `trace.parent()`), waits
for the card, sends the request as the task's data part, and stops the process after. The task's artifacts carry the record,
the facts, the summary, the picture, the table, the script, the thoughts and the trail; the client writes them under the
artifact's folder and saves the artifact. A test installs the server's app in process with `set_app`.

From the command line, one real ask: `uv run python -m causal_agent.evals.explore students "the mean reading score by lunch, as a picture"`.

## 6. The sandbox

`EXPLORE_SANDBOX` picks it ([common/config.py](../causal_agent/common/config.py)). `auto`, the default, becomes the strongest fence
the machine has, by code ([explore/sandbox.py](../causal_agent/explore/sandbox.py), `resolve`); a kind named and not at hand is
refused, never downgraded. The artifact records the kind that ran it.

- **seatbelt** (macOS, `sandbox-exec`): the network denied; writes denied outside the step's folder and the temp folders; reads
  under the home folder denied except the folder, the table, the interpreter and its base.
- **bwrap** (Linux): a new network namespace with nothing in it, the file system bound read-only but the folder.
- **unshare** (Linux, `unshare -rn`): a new network namespace with nothing in it.
- **subprocess**: the same interpreter with `-I`, the folder as the working directory, and an environment scrubbed to `PATH`,
  `HOME`, `DATA_CSV`, `MPLBACKEND=Agg` and `MPLCONFIGDIR`. Isolation by environment only; the fallback when no fence is at hand.
- **docker**: `docker run --rm --network none --memory 1g --cpus 1 --user uid:gid`, the table mounted read-only at `/data/table.csv`,
  the folder at `/work`, the image from `make explore-image` ([docker/explore.Dockerfile](../docker/explore.Dockerfile)): pandas,
  numpy, scipy, statsmodels and matplotlib, nothing else. No runtime, no run.

Every kind has a timeout, hands back what the script printed, and sweeps everything but `code.py`, `figure.png`, `facts.json`
and the small tables, number files, pictures and notes the script wrote.

## 7. Where an artifact lives and how it is cited

`data/memory/<name>/artifacts/<id>/`: `artifact.json`, `facts.json`, `code.py`, `thoughts.json`, `request.json`, `figure.png`
when it drew, the table file when it wrote one, and the `step_<n>/` folders. An artifact is tagged with the design run it is
about, or none; it is deleted with its dataset and never read back by code. The journal records an `explore` step naming the folder.

The addresses are `artifact:<id>`, `artifact:<id>.<fact>` for each number, `artifact:<id>.table` and `artifact:<id>.figure`.
Before the run and after it, the material lists every artifact about the data and the ones about the run in play
([desk/material.py](../causal_agent/desk/material.py), `add_artifacts`), so the Explainer can cite a number and the gate checks it
within one percent. The server serves any file an artifact lists at `GET /api/artifacts/{name}/{id}/{file}`
([server/artifacts.py](../causal_agent/server/artifacts.py)); the page shows the picture, the facts and the table link in
[Picture.tsx](../web/src/components/Picture.tsx).

## 8. A new version of the table

The same agent, asked for a table instead of a look (`ExploreRequest.product = "table"`; the desk's `transform` turn kind, read
like the others), writes a new version of the whole file rather than a table of facts. The projection then puts a row id
(`__row`) first; the agent must keep it on every row it keeps, may drop rows, may add, drop or recode columns, and must keep the
outcome and the treatment. The gate (`explore/gates.py::new_table_errors`) refuses a table that loses the row id, invents a row,
drops the pair or is empty. The client (`explore/client.py::transform`) re-joins the column held back by row id, restores the
file's column order, and saves the version with its script and record under `data/memory/<name>/versions/<v>/`
([memory/versions.py](../causal_agent/memory/versions.py)): `version:<v>`, its parent, what was asked, what changed. Nothing is
written in place. The desk puts the version in play on the person's word (`Memory.switch_version`: the file's facts and columns
re-read, a dropped column's fields kept out of play, the memory version moved, a `version` step in the journal) and shows the
design again on it before anything runs; after a run the same ask goes back to the ready moment on the new version. The
hand-off names the version's table and its number, so designs on different versions can be told apart.

```bash
uv run python -m causal_agent.evals.explore students "add a column that is 1 when reading score is 70 or more" --table
```

## Run figures are a different thing

| | run figure | artifact |
|---|---|---|
| what | data, never an image: a `FigureSpec` the browser draws | a picture or a table made by model-written code |
| who | lane code, deterministic, no model | the exploration agent plus the sandbox |
| checked by | `check_spec`: every `draws_on` address resolves in the run | the gates above |
| lives | `<run_dir>/figures.json`, copied to the design folder | `data/memory/<name>/artifacts/<id>/` |
| address | `figure:<id>`, `figure:<id>.<series>.<i>` | `artifact:<id>`, `artifact:<id>.<fact>`, `.table`, `.figure` |
