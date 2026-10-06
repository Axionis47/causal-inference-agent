# The exploration agent

A look at the data on request, from any turn of the conversation, before or after the run. The model that reads each message
decides that the person asked to look at something; a separate agent, in its own process, writes scripts, runs them in a
sandbox one step at a time, and answers one record; code gates the record; what it made lands on the artifacts shelf with an
address the chat can cite. It describes the data. It never estimates the effect.

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
swaps the dataset part for the pack's own rendering and appends the run's material lines that start with `design.`, `estimate:`,
`check:`, `refute:` or `placebo:`, so a number the agent quotes about the run is one the run holds.

**The no-join rule for free-form code.** The lane's tools refuse a join of the outcome with the treatment; a script may do
anything with the table it reads. So before a design has run, the table sent to the agent has one of the pair taken out, by code
([explore/project.py](../causal_agent/explore/project.py)): the outcome, unless the ask names the outcome and not the treatment,
then the treatment. An ask naming both is declined before any model call. The column held back is removed from the columns the
agent is told of, the prompt says it is held back, and a script that names it is refused. After a run nothing is held back.

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

## Run figures are a different thing

| | run figure | artifact |
|---|---|---|
| what | data, never an image: a `FigureSpec` the browser draws | a picture or a table made by model-written code |
| who | lane code, deterministic, no model | the exploration agent plus the sandbox |
| checked by | `check_spec`: every `draws_on` address resolves in the run | the gates above |
| lives | `<run_dir>/figures.json`, copied to the design folder | `data/memory/<name>/artifacts/<id>/` |
| address | `figure:<id>`, `figure:<id>.<series>.<i>` | `artifact:<id>`, `artifact:<id>.<fact>`, `.table`, `.figure` |

Decided in [ADR 0013](adr/0013-the-chat-is-the-thing-and-the-memory-has-four-shelves.md); the rule that no figure is drawn
unasked is [ADR 0007](adr/0007-a-picture-is-drawn-on-request.md).
