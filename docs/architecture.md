# How the code is laid out

One CSV and one question in; one run of one analysis family out, with a conversation before and after. The code is
organised by component: each family is one package, and the core packages beneath them know families only through a
registry. `import-linter` holds the layers and the rule that the core names no family; CI runs it on every push.

## The layers

Bottom to top. A package imports only what is below it.

| package | holds | may import |
|---|---|---|
| `common` | the contracts (the pack, the artifacts, the frame, the `Design` base), the address grammar, the model wrapper and its retry policy, the one config reading of the environment, the cite rule | nothing internal |
| `profile` | the deterministic profile of a CSV, the dataset index, the csv path rule | common |
| `memory` | what is known about a dataset: the field map, the store on disk, the gated write path, the checks, the shared probe helpers, the fit grid, the text views | common, profile |
| `viz` | figures as data with addresses: `FigureSpec`, the figure-pick graph, the post-run figures every lane can draw, the register a family puts its pre-run figures in | common, profile, memory |
| `lane` | the harness every lane is built on: intake, case, verify, asks, records, figures, words, the shared nodes, graph compile, task types, prompts, knowledge loader | common, profile, memory, viz |
| `families` | one package per family (below), the declared families, the registry | everything above |
| `desk` | the conversation graph, the routing, the pack builder, the material and the brief, the chat after a run | everything above |
| `server` | the FastAPI shell: datasets, sessions, the transcript, the projection into view models | everything above |
| `evals` | the eval runner and the command-line runs; outside the layers | anything |

The contract in `pyproject.toml` under `[tool.importlinter]` is the executable form of this table. `causal_agent/families/tests/test_core_names_no_family.py`
greps the five core packages for any family or engine name.

## A family package

```
families/<name>/
  family.yaml     what it answers, what it needs, what it assumes, and under needs_claims the claims the fit grid checks
  design.py       the family block: a Design subclass, registered by kind so a pack read from JSON comes back as it
  handoff.py      design_block(BlockInputs) -> Design: how the desk fills the block, by code, never as a constraint
  probes.py       probes(df, table, thresholds) -> list[ProbeResult]: what strikes the family out, computed once assignment is settled
  previz.py       the pre-run figures: each a FigureDecl the viz judgement reads beside the function that draws it
  postviz.py      the figures the lane draws from its own artifacts
  lane/           the engine and its judgements: nodes, graph, state, prompts, contracts, adapter, checks, shape, knowledge/*.yaml
  evals/          cases.yaml, the stored hand-offs, the summariser, the family's own evaluators, one EvalSpec
  tests/          the lane's suite and the figure tests
  __init__.py     FAMILY = FamilyDef(...): the one object the registry lists
```

`families/registry.py` lists the built families and the declared ones in the order the routing shows them. A declared
family (`families/declared/`) has a yaml, perhaps a probe, and the stub lane that says it is not supported yet. The desk
reaches a lane, a block, a probe or a figure only through `REGISTRY`; `memory.ops.probe` and `fit` take the families as
an argument; `viz.graph` chooses among the figures the point's family registered at import.

Adding a family means adding one package and one line in the registry. Nothing in the core changes.

## What the lanes share

`causal_agent/lane/` is the harness. A lane's own judgements stay in its package; what every lane copies is here once:
the pack weighed by code (`case`), the honest stop and the ask-back, the recorded declines, the figure tail, the node
plumbing (`nodes`), how a graph is compiled (`graph`), the relate and interpret task types (`state`), the pick and
interpret prompts and the plain-words rule (`prompts`), and the knowledge loader (`knowledge`). See
[lane-harness.md](lane-harness.md) for the plan that built it and [adr/0001-lanes-stay-distinct.md](adr/0001-lanes-stay-distinct.md)
for why a lane keeps its own engine.

## The three stores at run time

| store | where | written by | read by |
|---|---|---|---|
| the memory | `data/memory/<name>/` (`meta.yaml`, `columns.yaml`, `fields.yaml`, `said.jsonl`, `designs/<n>/`) | `memory.store.save`, through `memory.ops.apply` only | the desk, the pack builder, the viz tool |
| the checkpoints | `.artifacts/web/checkpoints.sqlite` | the desk graph through `SqliteSaver`, with the allowlist in `desk/graph.py` | the server's session manager |
| the run artifacts | `.artifacts/runs/<dataset>-<tag>-<id>/` (table, design, estimates, report, figures.json) | a lane, through `lane.records` | the desk's brief and the server's file routes |

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
`profile`, `server` and the family packages outside their lanes; the desk and the lanes are still `ignore_errors`, a
gap recorded in the plan.

## Decisions

The records under [adr/](adr/) hold the decisions the layout rests on. Add one when a decision of that weight is made.
