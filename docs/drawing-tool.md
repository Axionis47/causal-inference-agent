# The drawing tool

A picture of anything in the file, on request, before or after the run. The model writes one script; a sandbox runs it; the
numbers the picture shows come back as facts the chat can cite. The tool is never a "tool call" in the model's sense and never
a keyword match: the model that reads each message decides that the person asked to see something drawn.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="diagrams/draw-sequence-dark.svg">
  <img alt="The drawing tool as a sequence: the person, the Reader or Explainer, the desk's draw node, the drawing model, the sandbox, the store; two judgements and two gates" src="diagrams/draw-sequence-light.svg">
</picture>

## 1. How a request is detected

Two judgements can raise it, neither of them the drawing model.

**Before the run**, the Reader reads every message into a `Reading` ([desk/contracts.py:33](../causal_agent/desk/contracts.py)).
One field is `draw`: "what the person asked to see drawn, in their words; null when the message asks for none. A drawing request
is never an update." The Reader's system prompt says so in one sentence ([desk/prompts/journey.py](../causal_agent/desk/prompts/journey.py)).
The `infer` node keeps the request across retries and routes to the `draw` node ([desk/nodes/interview.py:488](../causal_agent/desk/nodes/interview.py)).
A drawing request wins over a question to the desk; when both arrive, `draw` goes on to `explain`.

**After the run**, the Explainer's reply is an `AfterReply` whose `kind` is one of answer, revise, what_if, requestion, draw, done
([desk/contracts.py:66](../causal_agent/desk/contracts.py)). A `draw` reply with an empty `draw` field is refused by the gate
([desk/explainer.py:164](../causal_agent/desk/explainer.py)). The `turn` node maps it to `draw_after` ([desk/nodes/after.py:163](../causal_agent/desk/nodes/after.py)).

The Explainer can also raise it before the run, since `answer` and `draw` are the only two kinds legal there.

## 2. What the drawing model is told

`draw_context` ([desk/nodes/interview.py:498](../causal_agent/desk/nodes/interview.py)) builds the context from the memory's text
view: the dataset, the change and the assignment, the beliefs, then one line per non-constant column. After a run, `draw_after`
swaps the dataset part for the pack's own rendering and appends the run's material lines that start with `design.`, `estimate:`,
`check:`, `refute:` or `placebo:` ([desk/nodes/after.py:258](../causal_agent/desk/nodes/after.py)).

The system prompt, in full ([viz/draw.py:25](../causal_agent/viz/draw.py)):

> You write one complete Python script that draws one figure from a CSV file, and nothing else.
>
> The script:
> - reads the CSV from the path in the environment variable VIZ_CSV with pandas
> - draws one matplotlib figure with the Agg backend and saves it as figure.png in the current directory
> - writes facts.json in the current directory: a JSON object from fact name to number, with every number the picture shows
>   (a count, a share, a mean, a difference), so the reader can quote each one
> - uses no network, writes no other file, and reads nothing but VIZ_CSV
> - names columns by the file's own names, exactly as listed
>
> Answer with the plan in one or two sentences, then the script alone in its own field with no words before or after it, a
> caption of one sentence in the person's words on what the picture shows, and the names of the numbers the script writes
> to facts.json.

The user prompt carries the ask, the context, and the columns as `key -> name` lines. No method and no column is named in the
prompt text itself; all of it arrives as data.

## 3. The contract and the gate

| step | what | where |
|---|---|---|
| in | `DrawRequest`: dataset, moment (pre or post), design, memory version, the ask, the context, the CSV path, the columns | [viz/draw.py:52](../causal_agent/viz/draw.py) |
| judgement | `DrawCode`: plan, code, caption, the fact names it promises | [viz/draw.py:63](../causal_agent/viz/draw.py) |
| gate 1 | the code parses (`ast.parse`); a syntax error retries without touching the sandbox | [viz/draw.py:114](../causal_agent/viz/draw.py) |
| run | `sandbox.run(code, csv, out_dir)` | [viz/sandbox.py:29](../causal_agent/viz/sandbox.py) |
| gate 2 | `figure.png` is non-empty; `facts.json` is a JSON object; every value is a number, no booleans; every promised name is present | [viz/draw.py:124](../causal_agent/viz/draw.py) |
| out | an `Artifact`, or after three failures a `Decline(check="draw.failed")` and no folder | [viz/store.py:25](../causal_agent/viz/store.py), [viz/draw.py:111](../causal_agent/viz/draw.py) |

A failed try's stderr, cut to its last 1500 characters, goes into the next prompt under "PREVIOUS ATTEMPT FAILED. Fix the script."

## 4. The sandbox

`VIZ_SANDBOX` picks it ([common/config.py:42](../causal_agent/common/config.py)). `auto`, the default, becomes the strongest fence
the machine has, by code ([viz/sandbox.py](../causal_agent/viz/sandbox.py), `resolve`); a kind named and not at hand is refused,
never downgraded. The artifact records the kind that ran it.

- **seatbelt** (macOS, `sandbox-exec`): the network denied; writes denied outside the artifact folder and the temp folders; reads
  under the home folder denied except the artifact folder, the CSV, the interpreter and its base. The folders between the home
  folder and those paths may be named, never listed. The rules are written per run, the last matching rule wins.
- **bwrap** (Linux): a new network namespace with nothing in it, the file system bound read-only but the artifact folder.
- **unshare** (Linux, `unshare -rn`): a new network namespace with nothing in it.
- **subprocess**: the same interpreter with `-I`, the artifact folder as the working directory, and an environment scrubbed to
  `PATH`, `HOME`, `VIZ_CSV`, `MPLBACKEND=Agg` and `MPLCONFIGDIR`. Isolation by environment only; the fallback when no fence is at hand.
- **docker**: `docker run --rm --network none --memory 1g --cpus 1 --user uid:gid`, the CSV mounted read-only at `/data/table.csv`,
  the folder at `/work`, the image from `make viz-image` ([docker/viz.Dockerfile](../docker/viz.Dockerfile)). No runtime, no picture.

Every kind has a 60 second timeout and sweeps everything but `code.py`, `figure.png` and `facts.json`. The tests run a script
that opens a socket, lists the home folder and writes into it under each fence the machine has, and expect all three refused.

## 5. Where a picture lives and how it is cited

| drawn | folder |
|---|---|
| before any run | `data/memory/<name>/viz/pre/<id>/` |
| after design n | `data/memory/<name>/designs/<n>/viz/<id>/` |

Each folder holds `code.py`, `figure.png`, `facts.json`, `request.json` and `artifact.json`. A picture is deleted with its dataset
or its design and is never read back by code. The journal records an `explore` step naming the folder.

The addresses are `artifact:<id>` for the picture and `artifact:<id>.<fact>` for each number. Before the run and after it, the
material lists every picture drawn so far with its numbers through one helper ([desk/material.py](../causal_agent/desk/material.py),
`add_artifacts`), so the Explainer can cite a drawn number and the gate checks it within one percent; an answer that states a number
the picture does not show is refused, and after three refusals the desk falls back.

The server serves the PNG at `GET /api/artifacts/{name}/{moment}/{design}/{id}/figure.png` ([server/artifacts.py](../causal_agent/server/artifacts.py));
the page shows it with its caption and its facts in [Picture.tsx](../web/src/components/Picture.tsx).

## Run figures are a different thing

| | run figure | drawn picture |
|---|---|---|
| what | data, never an image: a `FigureSpec` the browser draws | a PNG made by model-written code |
| who | lane code, deterministic, no model | the drawing model plus the sandbox |
| checked by | `check_spec`: every `draws_on` address resolves in the run | the two gates above |
| lives | `<run_dir>/figures.json`, copied to the design folder | the folders above |
| address | `figure:<id>`, `figure:<id>.<series>.<i>` | `artifact:<id>`, `artifact:<id>.<fact>` |

The adjustment lane's own figures are in [families/adjustment/postviz.py](../causal_agent/families/adjustment/postviz.py): the graph,
the balance per contrast, the effect against its refutations. See [viz/spec.py](../causal_agent/viz/spec.py) and [lane/figures.py](../causal_agent/lane/figures.py).

Decided in [ADR 0006](adr/0006-the-desk-operates-tools-reason-the-memory-escalates.md) and [ADR 0007](adr/0007-a-picture-is-drawn-on-request.md).
