# Testing without a model

No test calls a model. Every judgement is scripted, every store is pointed at a temporary directory, and the suite runs in CI
with no credentials. This is what lets a gate be tested as a gate: the fake returns a bad answer on purpose and the test asserts
the refusal.

## The fake

`DeskFake` ([desk/tests/fakes.py:169](../causal_agent/desk/tests/fakes.py)) stands in for the model wrapper. A test installs it with
`set_llm(fake)` and `set_llm(None)` restores the real one. Its `with_structured_output(schema)` returns an object whose `invoke`
hands the human prompt to `answer(schema, human)`, and `answer` dispatches on the schema the node asked for:

| schema asked | the fake answers with |
|---|---|
| `Reading` over a note | the students note's drafts |
| `Reading` over a message | an optional scripted `infer` callable, else `read_by_rule`: the story's updates, confirms on "yes", unknown on "don't know", and `address = value` lines |
| `QuestionFrame` | parsed from the QUESTION line: "how many" is not causal, "why" is a root cause, otherwise the students frame |
| `FamilyDecision` | adjustment, citing what it was given |
| `DesignBrief` | a queued brief, else `brief_by_rule` from the family block in the prompt |
| `AfterReply` | popped from the `explain` queue before the run and the `after` queue after it |
| `DrawCode` | a queued script, else one that draws mean math score by lunch and keeps two numbers |

It records every call and every prompt, so a test can assert what a node was given. `answer_ask(payload)` scripts the person's side
of the interview, answering each interrupt payload, so a whole conversation runs from question to done in one test.

Each lane has its own `FakeLLM` beside its tests, and the lanes' tests run the real engines, DoWhy, pyfixest, rdrobust, on the real
fixture files under `data/`.

## Where a test writes

Never under `data/`. The memory is held in process and the stores are pointed at `tmp_path`: `store.memory_for`, `store.snapshot`,
`J.open_journal` and `config.ROOT` are the seams. The fixture datasets under `data/` and their claims files are data the tests read,
not the past.

Set `LANGSMITH_TRACING=false` when running locally; `.env` turns tracing on and the tracer floods the output when the quota is spent.

```bash
make test
```

```bash
uv run pytest causal_agent/desk/tests/test_journey.py -k picture -q
```

The journey tests ([desk/tests/test_journey.py](../causal_agent/desk/tests/test_journey.py)) are the spine: the first turn asks the
question and refuses one the file cannot answer; the story is asked once, read back, then the gaps asked by decision; a brief that
cites nothing real is refused three times then falls back; the matrix is a record with a fit step only when a cell moves; a lane
that asks back gets its answer and runs again; a revision after the run goes back through the gate and the checks; a what-if runs on
a copy and leaves the memory alone.

## The evals

The evals are the only place a model runs on purpose. One runner for every family, each family keeping its own cases:

| piece | where |
|---|---|
| the spec: family, LangSmith dataset, cases dir, the lane, the summariser, the evaluators | [evals/spec.py](../causal_agent/evals/spec.py), one `SPEC` per family under `families/<name>/evals/` |
| the cases | `families/<name>/evals/cases.yaml`: an id, a dataset, a question, expected values that grade the design and the stop, not the exact effect |
| stored packs | `families/<name>/evals/handoffs/*.json`, for a case that must run the lane alone so a routing change cannot mask it |
| the shared evaluators | [evals/evaluators.py](../causal_agent/evals/evaluators.py): status match, estimator allowed, sign match, flags contain, stopped at, placebo ran, thoughts present |
| one run from the command line | [evals/lane.py](../causal_agent/evals/lane.py) |
| the LangSmith run | [evals/run.py](../causal_agent/evals/run.py), `make evals FAMILY=adjustment` |

A case from the adjustment file:

```yaml
- id: students_prep_course
  dataset: students
  question: Did completing the prep course raise math scores?
  expected:
    status: done
    adjustment_set_contains: [lunch, parental_level_of_education]
    not_in_graph: [reading_score, writing_score]
    estimator_in: [propensity_score_stratification, propensity_score_weighting, propensity_score_matching, linear_regression]
    effect_sign: {completed_vs_none: positive}
    placebo_passes: true
```

## What CI runs

`make check`: ruff (lint and format), the import contracts, mypy, pytest, eslint, prettier, tsc, vitest, the web build, and the
schema freshness. The layers are held by `import-linter` in `pyproject.toml`; a test greps the five core packages for any family or
engine name; the OpenAPI schema and the generated page types must match what the server declares, or the build fails.
