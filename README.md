# Causal desk

One CSV and one question of the form "what did this change do to that outcome". The desk asks the question first, validates it
against the file, asks one thing per turn until a family of analysis stands, runs that family's lane on a context pack, and talks
about the result in the person's own words.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/flow-dark.svg">
  <img alt="The flow: the question, the interview, the memory and the matrix, the design brief, the lane, the chat after; under each stage the model calls it makes, every one through a gate" src="docs/diagrams/flow-light.svg">
</picture>

Three families are built: adjustment (DoWhy), diff-in-diff (pyfixest), discontinuity (rdrobust). Four more are declared so the
routing can rule them in or out.

## Why it is built this way

The hard part of a causal question is not the estimator. It is understanding the problem: what happened, to whom, who decided,
and what else could explain the outcome. Get that clear and most of the design is no longer a judgement. It is a fact the data can
check or a rule code can apply. What is left, a handful of readings no check can settle, is where a model earns its place.

So the desk puts agency only where a judgement is required. Code computes every fact. The model answers closed questions, one at
a time, each gated by code that checks its citations and its numbers. The person's word is evidence that a check can confirm or
refute, never the truth by default. The memory of what is known outlives any conversation.

This is the workflow I understand best and can defend line by line. The problem statement exists to draw that line: clarity about
the problem is what separates the decisions that need judgement from the ones that do not.

## The stack

| part | built on |
|---|---|
| the desk graph, its interrupts and checkpoints | LangGraph, a sqlite checkpointer |
| the judgements | Gemini 2.5 Flash on Vertex AI through LangChain, temperature zero, structured output into Pydantic schemas, gated by code |
| the memory, the claim catalogue, the matrix | a YAML catalogue of claim kinds, Pydantic fields with provenance, YAML and JSON stores on disk |
| the lanes | DoWhy, pyfixest, rdrobust; one subprocess per run; a shared harness |
| the drawing tool | a model-written matplotlib script, run in a subprocess or a Docker container, its numbers kept as facts |
| the API and the page | FastAPI; an OpenAPI schema generated from the response models; React and Vite with types generated from the schema |
| quality | ruff, mypy, import-linter for the layers, pytest with a scripted model, eslint, vitest, LangSmith for traces and evals |

## How the reasoning works

**The matrix decides the family.** Every family says which claims it needs and which values fit. As the interview settles claims,
code fills a grid of families against claim kinds; the first cell that does not fit strikes a family; ready means nothing a
surviving family needs is unknown. [docs/memory-and-matrix.md](docs/memory-and-matrix.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/matrix-diff-dark.svg">
  <img alt="The matrix before and after three answers" src="docs/diagrams/matrix-diff-light.svg">
</picture>

**A lane is code with five questions in it.** The adjustment lane runs fourteen stages. Code decides every route. The model is
asked five closed questions, one column at a time where it matters, and every answer goes through a gate before the graph takes
it. [docs/lanes.md](docs/lanes.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/lane-swimlane-dark.svg">
  <img alt="The adjustment lane as a swimlane" src="docs/diagrams/lane-swimlane-light.svg">
</picture>

**Every judgement is one closed question, cited and checked.** Here is one from a real run: what the model was given, what came
back, and the six checks code ran on it. [docs/gates.md](docs/gates.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/judgement-relate-dark.svg">
  <img alt="One relate judgement opened up" src="docs/diagrams/judgement-relate-light.svg">
</picture>

## Known limits

- No stage holds the whole story. The lane relates one column at a time and the frame it reads is the question, not the context,
  so it cannot notice what a person notices across columns. Checkability per step was traded for that.
- The default sandbox isolates the drawing script by environment only; the container mode is the one that blocks the network.
- Before the run, the chat shows a drawn picture but cannot yet quote its numbers; after the run it can.

## See it

One real run, start to finish, with screenshots: [docs/demo.md](docs/demo.md). The records it left are under `docs/demo/`.

## Run it

```bash
make dev-api
```

```bash
make dev-web
```

The API on 8000, the page on 5173 proxying `/api`. Copy `.env.example` to `.env`: the model runs on Vertex AI with your gcloud
application default credentials; only LangSmith needs a key, and only for tracing.

One run from the command line:

```bash
uv run python -m causal_agent.evals.lane adjustment students "Did completing the prep course raise math scores?"
```

## Check it

```bash
make check
```

Everything CI runs: lint, types, tests, the web, the schema. `make` alone lists the targets.

## Read about it

- [docs/README.md](docs/README.md): one page per mechanism: the desk, the memory and the matrix, the gates, the pack and the
  addresses, the lanes, the drawing tool, testing, the page.
- [docs/architecture.md](docs/architecture.md): the layers, a family package, the stores, the wire, the checks.
- [docs/adr/](docs/adr/): the decisions the layout rests on.
- [causal_agent/README.md](causal_agent/README.md): the packages; each has its own README.
