# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Project rules

## No past to support

The code supports no earlier layout, name, checkpoint, or artifact.

- When a shape changes, the old shape is deleted, not read. No loader, branch, or fallback for "a file written before".
- Old analyses are deleted, not migrated: designs, threads, checkpoints, run dirs.
- No alias for a renamed route, class, function, or state key.
- A test that only proves an old shape still loads is deleted with the shape.
- The fixture datasets under `data/` and their claims files are data the tests use, not the past. They stay, and a
  field they spell is renamed in the file, not mapped in code.

## A real run is the bar

The scripted-model suite is the floor. A change to a judgement, a gate, a prompt, a rung or the desk is done only after
Gemini has run through it and the trace reads right. Do not add fakes or scripted tests as the check for such a change;
keep the suite green with the smallest touch, run a real analysis, and read it back:

```bash
uv run python -m causal_agent.evals.lane adjustment --handoff causal_agent/families/adjustment/evals/handoffs/gov_transfers_forced.json   # one lane alone, ~1 min
uv run python -m causal_agent.evals.lane adjustment students "Did completing the prep course raise math scores?"                          # routed, then the lane
make trace                      # the newest LangSmith trace as a tree, gate errors inline; ARGS="--thoughts" adds the model's thoughts, ARGS="--last 8" lists roots
```

A full conversation runs against `make dev-api` (upload, question, interview, `run`, chat, `done`). Report what the model
did, not only that tests pass. The owner decides the cost.

## The trace mirrors the graph

What LangSmith shows is the workflow as it ran: every judgement with its prompt, record, thoughts and gate; every tool call;
every file left behind; the lane under the desk's `run` node. That holds because the tracing sits in chokepoints a new piece
passes through anyway, and `causal_agent/common/tests/test_trace_policy.py` fails when one is bypassed.

- A model call goes through `common/llm.py::structured` or `lane/episode.py::run_episode`; a decision-model call through
  `common/llm.py::decide`; nothing else invokes a model.
- A graph is compiled with a `name=`; a root run is never called "LangGraph".
- A new process gets the parent run and the thread from `common/trace.py::child_env` and opens under `trace.parent()`;
  a subprocess is never a new root.
- A file written under a design folder, a run folder or a viz folder goes through `trace.write`, so it is attached.
- A refusal that re-prompts the model goes through `trace.gate`; a retry has a visible reason.
- A new writer or caller that may not use the chokepoint is added to the test's allowlist with the reason beside it.
- A span is opened around a call, never set as a config on a runnable: the test fakes have a bare `invoke` and no
  `with_config`. ADR 0012 has the rest.

The global writing and commit rules apply.

# Commands

Python runs through `uv run`; the page through `npm --prefix web`. `make` alone lists the targets.

```bash
make check                      # what CI runs, minus the web build: ruff, lint-imports, mypy, pytest, eslint, prettier, tsc, vitest, schema
make test                       # the Python suite (~7 min; the lanes run the real engines on the fixtures)
uv run pytest causal_agent/desk -q                                    # one package
uv run pytest causal_agent/desk/tests/test_journey.py -k picture -q   # one test
make lint types                 # ruff + import contracts + eslint; mypy + tsc
make fmt                        # ruff and prettier, in place
make schema                     # after any change to server/models.py: regenerates web/openapi.json and web/src/generated/schema.ts; CI fails on a diff
make dev-api                    # the API on :8000, the desk in process, the page served from web/dist when built
make dev-web                    # the page on :5173, proxying /api
make studio                     # langgraph dev on :2024: the desk and the adjustment lane from langgraph.json
make evals FAMILY=adjustment    # a family's cases against its LangSmith dataset; `evals.dataset <family>` uploads the cases first
make trace                      # the newest LangSmith trace as a tree (see the rule above)
make explore-image              # the exploration agent's container image, for EXPLORE_SANDBOX=docker
uv run python -m causal_agent.desk.handoff <dataset> --family <f> --outcome ... -o handoff.json   # a forced hand-off; scripts/forced_handoffs.sh regenerates the stored ones
uv run python -m causal_agent.desk.pipeline designs/<n>/handoff.json --json-file out.json         # the lane process exactly as the desk launches it
```

**Environment.** Copy `.env.example` to `.env`; `common/config.py` is the one reading of it. The model is Gemini on Vertex AI
through application-default credentials (`gcloud auth application-default login`); `VERTEX_PROJECT` falls back to the active
gcloud project. `INCLUDE_THOUGHTS` and `THINKING_BUDGET` are debug knobs. `RUN_DIR` and `PROFILE_CACHE_DIR` move the run
folders and the profile cache; `EXPLORE_SANDBOX` picks the fence a script runs in. `DECIDER` picks the decision model that reads each
turn's kind before any Gemini call: `laya` (the open checkpoint in this process; the first call loads it) or `jev` (a hosted
endpoint on the same wire, `DECIDER_URL` and `DECIDER_KEY`). `DECIDER_CHECKPOINT` names a fine-tune of the open checkpoint
that stands in for it; `make decider` trains one on `desk/evals/turns_train.jsonl` and scores it on the held-out
`desk/evals/turns.jsonl`, which `uv run python -m causal_agent.evals.turns` scores for any backend.

**Tests.** Set `LANGSMITH_TRACING=false` when running tests locally; `.env` turns tracing on and the tracer floods the output
when the quota is spent. Tests never call a model: every judgement is scripted through `set_llm(fake)`; see
`desk/tests/fakes.py`, the `FakeLLM` classes beside each lane's tests, and `docs/testing.md`. Tests never write under
`data/`: `conftest.py` points `RUN_DIR` and `PROFILE_CACHE_DIR` at `tmp_path` for every test, the `no_memory_writes`
fixture makes `store.save` a no-op, and `store.memory_for`, `store.snapshot`, `J.open_journal` and `config.ROOT` are the
seams a test points at a temporary directory. Tests that stand in for `desk.pipeline.run` take the same arguments it does,
`thread_id` included.

# Architecture

One CSV and one causal question in; one run of one analysis family out, with a conversation before and after. Read
`docs/architecture.md` (the components table and the layers), `docs/gates.md` (every model call, what it is given, what
code checks) and `docs/adr/` before changing structure. `docs/README.md` indexes the rest. ADR 0006 is the current shape:
the desk operates, tools reason, the memory escalates. ADR 0008 makes a judgement a bounded episode over a ladder, 0009 and
0011 declare a lane's design space, 0012 is the trace. ADR 0010 (a sealed rung is replayed, not rerun) is proposed, not built.

**Layers, bottom to top**, each importing only what is below, held by `import-linter` in `pyproject.toml`: `common`
(contracts, addresses, the model wrapper, the trace helpers, the config) → `profile` → `memory` → `viz` → `lane` (the
harness) → `families` (one package per family, reached only through `families/registry.py`) → `desk` → `server`; `evals`
is outside. A core package never names a family or an engine; `families/tests/test_core_names_no_family.py` greps for it.

**The desk** (`desk/graph.py`) is one LangGraph and the only graph: the question first, then one question per turn until
the matrix says ready, then the run, then the chat after. Nodes are code except the judgements, each one model call gated
by code (`infer`, `explain`, `read_question`/frame, `decide`, `design`, `judge`, `turn`, the exploration agent). Interrupts are
`ask_question`, `listen`, `talk`; the server resumes with `Command(resume=text)`. `listen` matches no word but quit: the Reader
reads every turn's kind, and a `run` read at the ready moment is what launches the analysis. `desk/route.py` is the same nodes as one function for the evals.

**The memory** (`memory/`) is what is known about a file: a map of `col:<key>.<field>` and `claim:<kind>.<field>` to a
`Field` with value, status, source, the sentence it rests on, and the reason. The catalogue of kinds is
`memory/fields.yaml`. The only write path is `ops.apply`: a model may only draft; only the person's word or a data check
confirms; a belief is written only on the person's word; the `story` kind is verbatim and written by the desk alone.
`ops.check` runs the data checks and the consistency rules; `ops.fit` computes the matrix (families × claim kinds) from each
family's `needs_claims`; ready means nothing required is open and one family survives. A family's decisions (`family.yaml`,
`rests_on`) also say what the interview asks beyond the required fields; those relations never block. `memory/facts.py`
computes data facts (`probe:data.*`) for the columns in play at every fit and at hand-off, never joining outcome and treatment.

**A lane** (`families/<name>/lane/`) runs in its own process on `designs/<n>/handoff.json` and reads nothing else
(`desk/pipeline.py`). Shared plumbing is in `lane/`; the engine, judgements and knowledge yaml stay in the family. A lane
climbs a ladder of rungs in the order an analyst reads its design (`Ladder` in the family's `contracts.py`, the shape in
`lane/ladder.py`); a rung is code where the pack settles it and a bounded episode where it does not (`lane/episode.py`: the
model may call the read-only tools in `lane/tools.py` within the budget its `checks.yaml` declares, then answers one typed
record that code gates; no tool joins the outcome with the treatment before the freeze). Every claim a judgement makes cites
an address the pack, the episode's facts or a rung below resolves; an unresolved cite is rejected and re-prompted. A rung
never asks the person; what it would not guess is a flag (`unsure`), and on a catalogue claim also a decline. The result is
`result.json` plus a run folder under `.artifacts/runs/`. The three built lanes are adjustment (DoWhy), diff-in-diff
(pyfixest) and discontinuity (rdrobust); four more families are declared so the routing can rule them in or out.

**The exploration agent** (`explore/`) looks at the data on request, from any turn, before or after a run: a separate agent
reached over A2A in its own process, one tool (`run_code`) that runs a script in the sandbox (`explore/sandbox.py`: the macOS
seatbelt, bwrap or unshare on Linux, Docker when asked, else a plain subprocess with a scrubbed environment), a bounded
episode over it, and one gated record naming the step whose files are the answer. Before a design has run the table it is
sent has the outcome or the treatment taken out, so it cannot estimate the effect. What it made lands on the artifacts shelf
(`memory/artifacts.py`), and the numbers in its `facts.json` become facts the chat can cite. Three tries, then a decline.

**Addresses** are the spine: `col:`, `claim:`, `probe:`, `check:`, `estimate:`, `refute:`/`placebo:`, `figure:`,
`artifact:`, `step:`, `user:turn:`. The chat after a run (`desk/material.py`, `desk/nodes/after.py`) may only say what it
can cite from the material, and every number it states must match its address within one percent.

**Stores on disk**: the memory under `data/memory/<name>/`; one folder per design run under `designs/<n>/` (snapshot,
pack, result, record, figures, drawn pictures under `viz/`); the journal under `analyses/<id>/journal.jsonl`; the
checkpoint sqlite and run folders under `.artifacts/`; the page transcript under `data/web/<name>/`. The memory outlives
conversations; graph state never is the source of truth.

**The wire**: `server/models.py` declares every response; the page types are generated from it, never typed by hand.
The page (`web/`, Vite + React) polls the session view and shows; it never decides.

**Adding a family** is one package (`family.yaml`, `design.py`, `handoff.py`, `probes.py`, `postviz.py`, `lane/`, `evals/`,
`tests/`) and one line in the registry; nothing in the core changes.
