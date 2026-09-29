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

The global writing and commit rules apply.

# Commands

Python runs through `uv run`; the page through `npm --prefix web`. `make` alone lists the targets.

```bash
make check                      # everything CI runs: ruff, lint-imports, mypy, pytest, eslint, prettier, tsc, vitest, web build, schema
make test                       # the Python suite
uv run pytest causal_agent/desk -q                                    # one package
uv run pytest causal_agent/desk/tests/test_journey.py -k picture -q   # one test
make lint types                 # ruff + import contracts + eslint; mypy + tsc
make fmt                        # ruff and prettier, in place
make schema                     # after any change to server/models.py: regenerates web/openapi.json and web/src/generated/schema.ts; CI fails on a diff
make dev-api                    # the API on :8000
make dev-web                    # the page on :5173, proxying /api
make viz-image                  # the drawing tool's container image, for VIZ_SANDBOX=docker
uv run python -m causal_agent.evals.lane adjustment students "Did completing the prep course raise math scores?"   # one routed run
```

Set `LANGSMITH_TRACING=false` when running tests locally; `.env` turns tracing on and the tracer floods the output when the
quota is spent. Tests never call a model: every judgement is scripted through `set_llm(fake)`; see `desk/tests/fakes.py`
and the `FakeLLM` classes beside each lane's tests. Tests never write under `data/`: the memory is held in process and the
stores are pointed at `tmp_path` (`store.memory_for`, `store.snapshot`, `J.open_journal`, `config.ROOT`).

# Architecture

One CSV and one causal question in; one run of one analysis family out, with a conversation before and after. Read
`docs/architecture.md` (the components table and the layers) and `docs/adr/` (the decisions) before changing structure.
ADR 0006 is the current shape: the desk operates, tools reason, the memory escalates.

**Layers, bottom to top**, each importing only what is below, held by `import-linter` in `pyproject.toml`: `common`
(contracts, addresses, the model wrapper) → `profile` → `memory` → `viz` → `lane` (the harness) → `families` (one package
per family, reached only through `families/registry.py`) → `desk` → `server`; `evals` is outside. A core package never
names a family or an engine; `families/tests/test_core_names_no_family.py` greps for it.

**The desk** (`desk/graph.py`) is one LangGraph and the only graph: the question first, then one question per turn until
the matrix says ready, then the run, then the chat after. Nodes are code except the judgements, each one model call gated
by code (`infer`, `explain`, `read_question`/frame, `decide`, `turn`, the drawing tool). Interrupts are `ask_question`,
`listen`, `talk`; the server resumes with `Command(resume=text)`. `desk/route.py` is the same nodes as one function for the
evals.

**The memory** (`memory/`) is what is known about a file: a map of `col:<key>.<field>` and `claim:<kind>.<field>` to a
`Field` with value, status, source, the sentence it rests on, and the reason. The catalogue of kinds is
`memory/fields.yaml`. The only write path is `ops.apply`: a model may only draft; only the person's word or a data check
confirms; a belief is written only on the person's word; the `story` kind is verbatim and written by the desk alone.
`ops.check` runs the data checks and the consistency rules; `ops.fit` computes the matrix (families × claim kinds) from each
family's `needs_claims`; ready means nothing required is open and one family survives. A family's decisions (`family.yaml`,
`rests_on`) also say what the interview asks beyond the required fields; those relations never block. `memory/facts.py`
computes data facts (`probe:data.*`) for the columns in play at every fit and at hand-off, never joining outcome and treatment.

**A lane** (`families/<name>/lane/`) runs in its own process on `designs/<n>/handoff.json` and reads nothing else
(`desk/pipeline.py`). Shared plumbing is in `lane/`; the engine, judgements and knowledge yaml stay in the family. Every
claim a judgement makes cites an address the pack resolves; an unresolved cite is rejected and re-prompted. The result is
`result.json` plus a run folder under `.artifacts/runs/`.

**Addresses** are the spine: `col:`, `claim:`, `probe:`, `check:`, `estimate:`, `refute:`/`placebo:`, `figure:`,
`artifact:`, `step:`, `user:turn:`. The chat after a run (`desk/material.py`, `desk/nodes/after.py`) may only say what it
can cite from the material, and every number it states must match its address within one percent.

**Stores on disk**: the memory under `data/memory/<name>/`; one folder per design run under `designs/<n>/` (snapshot,
pack, result, record, figures, drawn pictures under `viz/`); the journal under `analyses/<id>/journal.jsonl`; the
checkpoint sqlite and run folders under `.artifacts/`; the page transcript under `data/web/<name>/`. The memory outlives
conversations; graph state never is the source of truth.

**The wire**: `server/models.py` declares every response; the page types are generated from it, never typed by hand.

**Adding a family** is one package (`family.yaml`, `design.py`, `handoff.py`, `probes.py`, `postviz.py`, `lane/`, `evals/`,
`tests/`) and one line in the registry; nothing in the core changes.
