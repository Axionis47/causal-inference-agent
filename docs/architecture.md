# How the code is laid out

One CSV and one question in; one run of one analysis family out, with a conversation before and after. The code is
organised by component: each family is one package, and the core packages beneath them know families only through a
registry. `import-linter` holds the layers and the rule that the core names no family; CI runs it on every push.

## Components and what each owns

One responsibility per component, one contract in and one out, its own tests. [adr/0006](adr/0006-the-desk-operates-tools-reason-the-memory-escalates.md)
names the layers this table serves.

| component | owns | in | out |
|---|---|---|---|
| the desk (`desk/graph.py`, `desk/nodes/`) | the conversation: receive, classify, dispatch, gate, write, ask the next thing | a message | the next prompt, a journal step |
| the reader (`desk/nodes/frame.py` mine, `interview.py` infer) | words into claims with reasons; never a confirmation | a sentence or a note, the open fields | `ops.Update`s, drafted |
| the memory (`memory/`) | what is known: fields with status, source, sentence, reason; the one write gate; the checks; the matrix | updates | a field map, findings, the fit |
| the routing (`desk/nodes/decide.py`, `desk/route.py`) | which family stands, by code; a judgement only among several | the memory, the probes | a decision and its record |
| the pack builder (`desk/handoff.py`) | the one projection of the memory a lane sees | the memory, the frame, the decision | a `Handoff` |
| a lane (`families/<name>/lane/`) | the engine and its judgements on the pack alone, in its own process | `handoff.json` | `result.json`, the run folder |
| the drawing tool (`viz/draw.py`, `viz/sandbox.py`; what it makes lives on the artifacts shelf, `memory/artifacts.py`) | a picture from the file on request, with its code and numbers, in the strongest sandbox at hand | a `DrawRequest` | an `Artifact` or a `Decline` |
| the material and the brief (`desk/material.py`) | every artifact as a line with an address; the opening message after a run | a run record, the memory, the journal | citable lines |
| the chat after (`desk/nodes/after.py`) | one routed reply per message: answer, revise, what-if, requestion, draw, done | the material, the message | a gated reply |
| the server (`server/`) | the wire: sessions over the desk, the views, the files and pictures | HTTP | `SessionView` and files |
| the page (`web/`) | showing, never deciding | the wire | the screen |
| the trace (`common/trace.py`) | the span tree that mirrors the graph: a span per judgement, episode, try, tool call, gate and file; the parent handed to the lane's process | a call at a chokepoint | a LangSmith run under the node that made it |

## The layers

Bottom to top. A package imports only what is below it.

| package | holds | may import |
|---|---|---|
| `common` | the contracts (the pack, the artifacts, the frame, the `Design` base), the address grammar, the model wrapper and its retry policy, the trace helpers, the one config reading of the environment, the cite rule | nothing internal |
| `profile` | the deterministic profile of a CSV, the dataset index, the csv path rule | common |
| `memory` | what is known about a dataset: the field map, the store on disk, the gated write path, the checks, the shared probe helpers, the fit grid, the text views | common, profile |
| `viz` | pictures: `FigureSpec`, the figures a lane draws from its own artifacts (`postviz`); the drawing tool (`draw`, `sandbox`) and where what it draws lives (`store`) | common, profile, memory |
| `lane` | the harness every lane is built on: intake, case, verify, asks, records, figures, words, the episode and its tools, the ladder's shape, the shared nodes, graph compile, task types, prompts, knowledge loader | common, profile, memory, viz |
| `families` | one package per family (below), the declared families, the registry | everything above |
| `desk` | the conversation graph, the routing, the pack builder, the material and the brief, the chat after a run | everything above |
| `server` | the FastAPI shell: datasets, sessions, the transcript, the projection into view models | everything above |
| `evals` | the eval runner and the command-line runs; outside the layers | anything |

The contract in `pyproject.toml` under `[tool.importlinter]` is the executable form of this table. `causal_agent/families/tests/test_core_names_no_family.py`
greps the five core packages for any family or engine name.

## The trace

The LangSmith trace is the graph as it ran, and it stays that way when the graph changes because the tracing lives in the
chokepoints a new piece passes through anyway ([adr/0012](adr/0012-the-trace-mirrors-the-graph.md)):

| anything new that | goes through | and shows as |
|---|---|---|
| calls the model | `common/llm.py::structured` or `lane/episode.py::run_episode` | a span named for the judgement: prompt in, record and thoughts out; under an episode, a `try n`, a `look`, one tool span per call, a `gate` |
| compiles a graph | `.compile(name=...)`, via `desk/graph.py` or `lane/graph.py` | a named root (`desk`, `lane:<family>`); the server names each turn `<dataset> <analysis> turn <n> <phase>` |
| starts a process | `common/trace.py::child_env` and `trace.parent()` in the child (`desk/pipeline.py`) | the lane's graph under the desk's `run` node, on the same thread |
| writes under a design, run or viz folder | `common/trace.py::write` | an `artifact:<file>` tool span with the file attached |
| refuses a model answer | `common/trace.py::gate` | a `gate` tool span with the errors |
| appends a journal step or saves the memory | `Journal.append`, `store.save` | a `journal` or `memory.save` tool span |

`causal_agent/common/tests/test_trace_policy.py` greps for a bypass. `make trace` prints the newest trace as a tree.

## A family package

```
families/<name>/
  family.yaml     what it answers, what it needs, what it assumes, and under needs_claims the claims the fit grid checks
  design.py       the family block: a Design subclass, registered by kind so a pack read from JSON comes back as it
  handoff.py      design_block(BlockInputs) -> Design: how the desk fills the block, by code, never as a constraint
  probes.py       probes(df, table, thresholds) -> list[ProbeResult]: what strikes the family out, computed once assignment is settled
  postviz.py      the figures the lane draws from its own artifacts
  lane/           the engine and its judgements: nodes, graph, state, prompts, contracts, adapter, checks, shape, knowledge/*.yaml
  evals/          cases.yaml, the stored hand-offs, the summariser, the family's own evaluators, one EvalSpec
  tests/          the lane's suite and the postviz tests
  __init__.py     FAMILY = FamilyDef(...): the one object the registry lists
```

`families/registry.py` lists the built families and the declared ones in the order the routing shows them. A declared
family (`families/declared/`) has a yaml, perhaps a probe, and the stub lane that says it is not supported yet. The desk
reaches a lane, a block or a probe only through `REGISTRY`; `memory.ops.probe` and `fit` take the families as an argument.

Adding a family means adding one package and one line in the registry. Nothing in the core changes.

## What the lanes share

`causal_agent/lane/` is the harness. A lane's own judgements stay in its package; what every lane copies is here once:
the pack weighed by code (`case`), the honest stop and the ask-back, the recorded declines, the figure tail, the node
plumbing (`nodes`), the bounded episode a judgement may run as (`episode`) and the read-only data tools it may be offered
(`tools`), the shape every lane's ladder shares and the threats and flags every design carries (`ladder`), how a graph is
compiled (`graph`), the state keys and task types (`state`), the pick and interpret prompts and the plain-words rule
(`prompts`), and the knowledge loader (`knowledge`). See
[lanes.md](lanes.md) for the harness in use and [adr/0001-lanes-stay-distinct.md](adr/0001-lanes-stay-distinct.md)
for why a lane keeps its own engine.

## The six stores at run time

| store | where | written by | read by |
|---|---|---|---|
| the memory | `data/memory/<name>/` (`meta.yaml`, `columns.yaml`, `fields.yaml`, `said.jsonl`): what is known about the file; every field carries its status, its source, the sentence it rests on, and the reason | `memory.store.save`, through `memory.ops.apply` only | the desk, the pack builder, the drawing tool's context, the server's view |
| the designs | `data/memory/<name>/designs/<n>/`, one folder per design run: the memory snapshot, the pack, the frame, the decision, then the lane's result, the figures and the run record | `memory.store.snapshot` and `desk.nodes.run` for the design, the lane's process for `result.json`, `desk.pipeline.save_record` for the record | the lane, the brief, the server's runs list |
| the journal | `data/memory/<name>/analyses/<id>/journal.jsonl`: one conversation's steps, each with its address `step:<n>`, what it read, what it left, and the design run it belongs to | `memory.journal`, from the desk nodes as they run | the after-run chat and explain (citable lines), the server's Journal tab |
| the checkpoints | `.artifacts/web/checkpoints.sqlite`: where each thread's conversation is | the desk graph through `SqliteSaver`; `desk/graph.py` `_CONTRACTS` lists the classes a checkpoint may hold | the server's session manager |
| the run artifacts | `.artifacts/runs/<dataset>-<tag>-<id>/` (table, design, estimates, report, figures.json) | a lane, through `lane.records` | the desk's brief and the server's file routes |
| the pictures | `data/memory/<name>/viz/pre/<id>/` before any run, `data/memory/<name>/designs/<n>/viz/<id>/` after run n: `request.json`, `code.py`, `figure.png`, `facts.json`, `artifact.json` | the drawing tool, `viz.draw`, on the person's request | the chat (`artifact:<id>` and `artifact:<id>.<fact>`), the server's picture route; never code |

The memory is what is known, the journal is what was done, the checkpoint is where the conversation is. A conversation can be
thrown away and the knowledge survives; the knowledge can be revised and every design still shows what it was made from.

The transcript the page shows is `data/web/<name>/transcript.jsonl`; `meta.json` beside it holds the thread and the
last prompt. `common.config` reads every path and knob from the environment in one place.

## The wire

The server declares every response shape once in `server/models.py`; `FigureSpec` is the viz model itself. `make schema`
dumps the OpenAPI schema to `web/openapi.json`; `npm run types` generates `web/src/generated/schema.ts` from it;
`web/src/types.ts` is a page of aliases over that. CI regenerates both and fails on a diff, so a field the server renames
breaks the build, not the page.

## The checks

`make check` runs what CI runs: ruff (lint and format), the import contracts, mypy, pytest, eslint, prettier, tsc,
vitest, the web build, and the schema freshness. mypy is `check_untyped_defs` on `common`, `lane`, `viz`, `memory`,
`profile`, `server` and the family packages outside their lanes; the desk and the lanes are still `ignore_errors`, the
one typing gap left.

## Decisions

The records under [adr/](adr/) hold the decisions the layout rests on. Add one when a decision of that weight is made.

## The mechanisms, one page each

[desk.md](desk.md), [memory-and-matrix.md](memory-and-matrix.md), [gates.md](gates.md), [pack-and-addresses.md](pack-and-addresses.md),
[lanes.md](lanes.md), [drawing-tool.md](drawing-tool.md), [testing.md](testing.md), [page.md](page.md); the index is [README.md](README.md).
