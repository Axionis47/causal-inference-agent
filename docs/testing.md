# Testing policy

## Current trust status

**We do not yet trust the repository's test suite as a whole as evidence of correctness or production readiness.** Existing tests must be reviewed individually for what they actually prove. A green run, a large test count or high coverage is not sufficient evidence.

Some ingestion tests have been audited, repaired and checked against deliberate defects. That evidence applies only to the cases and boundaries exercised, not to the rest of the application. See the [ingestion credibility audit](audits/ingestion-test-credibility.md). Tests not yet reviewed remain unverified; this does not automatically mean they are wrong.

This policy sets the required organization and acceptance standard. It does not claim that the existing suite already complies. Apply it phase by phase rather than reorganizing or certifying the whole application at once.

## Tests follow code ownership and behavior

Keep tests beside the component that owns the behavior. Use the [component handbook](components/README.md) and its [source index](components/source-index.md) to identify that owner.

- Python component tests belong under the owning package's `tests/` directory.
- Frontend component and adapter tests belong beside their implementation as `*.test.ts` or `*.test.tsx`.
- Browser journeys belong under `web/e2e/`; deployment checks belong with deployment tooling.
- Place integration tests with the component that owns the boundary, identifying the real producers and consumers involved.
- Keep shared fixtures and test utilities in one appropriate shared location. Do not duplicate storage setup, fake implementations or assertions across phases.
- When production ownership, contracts or paths change, update the responsible tests and their consumers in the same change. Remove tests for deleted behavior; do not preserve obsolete compatibility paths.

Following the code means following its responsibilities and public behavior. It does not mean copying the implementation into assertions, creating a test for every private function or reproducing the exact file tree mechanically.

## Organize acceptance by user-journey phase

Each phase needs a verification map linking its requirements to the actual test files and cases. Keep that map in the responsible component chapter or phase verification document, without copying test implementations into documentation.

For each phase, record:

1. Its starting point, completion point and explicit exclusions.
2. The frontend, API, application logic and storage components responsible for it.
3. Inputs, outputs, contracts and invariants that must hold across their boundaries.
4. Tests for normal behavior, invalid input, relevant failures, recovery and concurrency.
5. Both write and read paths, including ownership and exact persistence where applicable.
6. The required commands and CI jobs, observed results and unresolved gaps.

The current ingestion phase ends at upload, validation, durable storage, confirmation and retrieval. Profiling, dataset review, conversations and analysis require their own phase maps and acceptance evidence. Shared boundaries are tested once at their owner and referenced by each consuming phase.

## Select the test layer that proves the requirement

| Layer | Required purpose |
|---|---|
| Unit | Check a local rule, parser or calculation with small explicit inputs and independently determined expectations. |
| Component | Check observable component behavior, including frontend state, errors and submitted actions. |
| Contract | Check that actual producers and consumers agree on schemas, payloads, errors and invariants. Generated-file freshness alone does not prove runtime behavior. |
| Integration | Exercise real boundaries. Use PostgreSQL and object storage for claims about their constraints, transactions, locks and persistence. |
| End-to-end | Exercise a representative phase through the running application and real browser, including relevant retrieval paths. |
| Deployment | Verify the built application, dependencies and persistence across actual process or container replacement. |
| Model acceptance | Exercise changed model judgments and causal workflows with real models and inspect their traces under the existing acceptance policy. |

Failure recovery, security, concurrency and performance are risks to cover at the relevant layers. They are not satisfied merely by adding a test with that label. Use the smallest layer that proves the requirement, with a limited set of complete journeys connecting the components.

## Credibility requirements for every test

Every test must have a clear setup, action and observable expected result. During review, identify the requirement it protects, the code exercised, the dependencies replaced and the defect that would make it fail.

- Assert the intended outcome and failure reason. An unrelated exception must not satisfy an ownership, validation or recovery assertion.
- Establish the necessary starting condition. To prove deletion, first prove the object exists. To prove persistence, read through a fresh relevant process or connection as required by the claim.
- Include meaningful positive and negative cases, exact limits and relevant side effects. A rejected operation must not silently publish partial state.
- Use mocks to control dependencies or inject failures, never as evidence that the replaced dependency works. Test expectations must not be calculated by the same implementation being tested.
- For critical safeguards, deliberately break or disable the behavior and verify that the intended assertion fails. Apply mutations only in disposable resources or temporary copies. Setup failures, teardown errors, skipped tests and unrelated failures do not count as defect detection.
- Reproduce a fixed bug with a failing regression test before accepting the repair. Do not weaken assertions just to obtain a green run.

Review tests individually and record their status as unreviewed, credible for a stated scope, or requiring repair. Report missing coverage explicitly. Do not add low-value tests solely to increase counts or a coverage percentage.

## Isolation and repeatability

Tests must own their temporary data and resources. They must not write to repository fixtures, use production storage or inherit deployment credentials and endpoints. Real storage tests need isolated databases and private buckets, real migration setup and teardown scoped only to their owned resources.

Tests must run independently of order and previous runs. Restore patched dependencies, close clients and stop child processes. Concurrency tests must create actual overlapping operations with bounded synchronization. Recovery tests must exercise the failure boundary they claim to cover.

Ordinary automated tests must not call a model. Real-model acceptance remains a separate, explicitly controlled workflow; scripted models cannot establish the quality of real judgments or replace required trace readback.

## CI and phase completion

Required phase checks must run in CI with their real dependencies. Missing infrastructure, empty test selection and skipped required cases must fail the required job. Optional local skips must be visible and must not be reported as acceptance.

Retain actionable failure evidence such as test reports, relevant logs and browser traces without exposing private data or secrets. Check generated contracts alongside runtime tests. Keep real-model acceptance separately controlled.

A phase is complete only when its verification map covers the scoped requirements, its credible tests and required checks pass, critical negative controls detect their intended defects, and outstanding limitations are recorded. A successful phase does not certify unrelated phases or the entire application.

## Existing scripted-test mechanics

The sections below describe the existing test and evaluation tools. Their presence is not a blanket credibility endorsement.

No test calls a model. Every judgement is scripted, every store is pointed at a temporary directory, and the suite runs in CI
with no credentials. This is what lets a gate be tested as a gate: the fake returns a bad answer on purpose and the test asserts
the refusal.

## The fake

`DeskFake` ([desk/tests/fakes.py:169](../causal_agent/desk/tests/fakes.py)) stands in for the model wrapper. A test installs it with
`set_llm(fake)` and `set_llm(None)` restores the real one. The decision model is swapped the same way with `set_decider(fake)`, and `DeskFake` is both: its `predict` answers by the cues `read_by_rule` reads and is unsure on anything else. Its `with_structured_output(schema)` returns an object whose `invoke`
hands the human prompt to `answer(schema, human)`, and `answer` dispatches on the schema the node asked for:

| schema asked | the fake answers with |
|---|---|
| `Reading` over a note | the students note's drafts |
| `Reading` over a message | an optional scripted `infer` callable, else `read_by_rule`: the kind by cue (a run word, a message starting with plot or show, one ending in a question mark), the story's updates, confirms on "yes", unknown on "don't know", and `address = value` lines |
| `QuestionFrame` | parsed from the QUESTION line: "how many" is not causal, "why" is a root cause, otherwise the students frame |
| `FamilyDecision` | adjustment, citing what it was given |
| `DesignBrief` | a queued brief, else `brief_by_rule` from the family block in the prompt |
| `AfterReply` | popped from the `explain` queue before the run and the `after` queue after it |
| `ExploreAction`, `RunCodeStep`, `Exploration` | the exploration agent's look (the script that draws mean math score by lunch, then done) and its record (two numbers and the picture), or a queued record |

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
make verify-chat
make verify-chat WITH_MODEL=1
```

The chat runner combines domain contracts, real PostgreSQL/private-storage tests, four deliberate defects, browser journeys, container replacement, contract freshness, lint/types, build and trace policy. Its real-model stages are reported separately. Evidence is retained under `.artifacts/chat-verification/<run>/`. Semantic review of questions, accepted edits, citations, tool calls and stopping behavior is required even when automation passes.

[Chat command tests](../causal_agent/server/tests/context/test_chat_commands.py) exercise the actual revision and operation boundaries, including pinning, duplicate submissions, ordinary/hypothetical no-writes, structured unknowns, pause/resume, invalidated authorization and interrupted recovery. [Snapshot tests](../causal_agent/memory/context/tests/test_snapshot.py) launch all three real lane intake boundaries. The old filesystem interview journeys and prose-choice parser tests have been removed with those paths.

## The evals

The evals are the only place a model runs on purpose. One runner for every family, each family keeping its own cases:

Analysis acceptance uses real-source datasets, including live Kaggle datasets uploaded through the ordinary intake path. The original example files under `data/raw/` remain available for regression checks. Generated evaluation datasets, their generators and dependent handoffs and reports have been removed. New cases must record the dataset's source and the supplied study account. A downloaded CSV alone does not establish the causal assumptions. Small local unit inputs still exercise parsing, contracts and numerical invariants; they do not count as real-data endpoint acceptance.

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

`make check`: ruff (lint and format), the import contracts, mypy, pytest, eslint, prettier, tsc, vitest, and schema/ladder generation.
The web build runs separately through `make build-web`. CI checks generated OpenAPI, TypeScript and ladder freshness.
The layers are held by `import-linter` in `pyproject.toml`; a test greps the core packages for any family or
engine name; the OpenAPI schema and the generated page types must match what the server declares, or the build fails.

The [production verification plan](production/verification.md) distinguishes implemented ingestion checks from proposed checks for private workspaces and CSV lineage. The dedicated ingestion CI job runs isolated real storage tests, deliberate-defect controls and the built-image browser/persistence smoke. Broader planned checks do not count as implemented evidence.

## Verification register

This register records what we currently trust for a specific scope, based on review and observed execution. It is not a claim that every test in the containing folder has been verified. Unlisted tests remain unreviewed under this policy, even if their latest run passed.

### Phase 1: Dataset ingestion

Phase 2 navigation and shared-fixture changes were re-verified locally on 23 October 2026. See the [Phase 2 evidence](audits/phase2-library.md) for the current runs; the earlier ingestion audit remains historical evidence.

Scope: upload form, validation, durable storage, confirmation and retrieval. All entries below belong to Phase 1. Profiling, dataset review, conversations and analysis are excluded.

Last verified locally: **23 October 2026**. Detailed results and limitations: [ingestion repair evidence](audits/ingestion-test-credibility.md#repair-evidence-23-october-2026).

| Verified location | Scope we trust | Evidence and limits |
|---|---|---|
| [server/tests/test_ingestion.py](../causal_agent/server/tests/test_ingestion.py) | CSV validation, original preservation, ownership, publication, retries, concurrency and cleanup for ingestion | 48 cases passed, including real isolated PostgreSQL/S3. Seven targeted backend defects were detected. Other files in `server/tests/` are not certified by this entry. |
| [web/src/api.test.ts](../web/src/api.test.ts) | Actual upload adapter's transmitted fields, bytes, progress and error handling | Five cases passed; three source mutations were detected. Transport is replaced in these tests; real browser transmission is covered separately below. |
| [web/src/pages/NewDataset.test.tsx](../web/src/pages/NewDataset.test.tsx) | Upload form state, retry keys, repeated-submit prevention and confirmation retrieval behavior | Three current rendered component cases passed after the confirmation view moved to the details page. API responses are mocked; this entry does not establish real persistence. |
| [web/e2e/ingestion.spec.ts](../web/e2e/ingestion.spec.ts) | Browser upload, confirmation, reload and exact original retrieval | The updated two-dataset journey passed against the built application with real API and storage. This row records ingestion coverage; the same journey now also exercises profiling, recorded under Phase 3. |
| [docker/smoke.py](../docker/smoke.py) | Current-image startup and ingestion persistence across replacement of app, database and object-store containers | Passed with all three container IDs changed and exact saved data retrieved. Does not establish backup restoration, load capacity or production authentication. |
| [server/tests/ingestion_resources.py](../causal_agent/server/tests/ingestion_resources.py), [backend controls](../scripts/ingestion_mutations.py) and [frontend controls](../scripts/ingestion_web_mutations.py) | Isolated ingestion test resources and rejection of false acceptance | Real storage runs used harness-owned resources; all ten deliberate defects were detected. Empty required selection and skipped required cases were correctly rejected. This is targeted evidence, not exhaustive fault coverage. |

The ingestion CI job is configured, but its hosted execution has not yet been observed. The passing full frontend and ordinary server runs do not extend this register to their other test files.

### Phase 2: Return to your datasets

Verified locally: **23 October 2026**. Scope: owned listing, canonical dataset details, original context and CSV retrieval. [Evidence and cleanup record](audits/phase2-library.md).

| Verified location | Scope we trust | Evidence and limits |
|---|---|---|
| [server/tests/library/](../causal_agent/server/tests/library/test_library.py) | Owner-filtered ready datasets, metadata contracts, stable pagination, cursor validation, read-only behavior and service errors | All 14 cases reviewed and passed with real isolated storage. Three targeted listing defects were detected. Development identity only. |
| [Landing.test.tsx](../web/src/pages/Landing.test.tsx) | Shared homepage/sidebar collection, loading/empty/errors, pagination retry and stale-response protection | Three component/hook tests passed. API is mocked; storage evidence is separate. |
| [DatasetDetails.test.tsx](../web/src/pages/DatasetDetails.test.tsx) | Exact plain-text context, download URL, retrieval retries and stale navigation responses | Three rendered tests passed; API is mocked. |
| [Browser journey](../web/e2e/ingestion.spec.ts) and [Compose smoke](../docker/smoke.py) | Two uploads, library return, refreshed details, exact downloads and list persistence across container replacement | Passed on the final built image with real PostgreSQL/S3. Shared evidence with Phase 1, not duplicate test implementations. |

The dedicated combined backend run passed all **62** cases without skips. All **58** frontend tests passed; this does not certify unreviewed frontend files. CI is configured for both phases; hosted execution remains unobserved. Run `make test-library` for this phase, `make test-ingestion` for both, `make test-ingestion-mutations` for negative controls, and `make container-smoke` for the browser/deployment journey. Docker is required for storage checks; install Chromium with `npm --prefix web exec -- playwright install chromium` before the local browser smoke.

### Phase 3: Read-only dataset and column profiling

**Milestone accepted with known limitations, 23 October 2026.** The user accepted deferring the two precision defects discovered after verification: extremely small numeric strings can underflow to zero, and large decimal-form numbers can silently lose precision. Those cases are not covered by the passing regression suite and are not certified correct. Maximum-width (200-column) performance also remains unverified. See the audit below for concrete examples. No runtime fixes or new tests are claimed by this acceptance.

Verified locally: **23 October 2026**. Profiles are owned, version-linked PostgreSQL records; original CSVs and context remain unchanged. [Implementation and evidence](audits/phase3-profiling.md).

| Verified location | Scope we trust | Evidence and limits |
|---|---|---|
| [profile/tests/test_inspection_columns.py](../causal_agent/profile/tests/test_inspection_columns.py), [dataset tests](../causal_agent/profile/tests/test_inspection_dataset.py), [failure tests](../causal_agent/profile/tests/test_inspection_failures.py) | Preserved field text, type suggestions, bounded measurements, candidate keys and visible failures | 29 new calculation cases passed. The full profile folder, including its seven existing cases, passed 36 tests. This does not establish arbitrary-format support. |
| [server/tests/profiling/](../causal_agent/server/tests/profiling/) | Owned APIs, column identity, owner/version FKs, atomic publication, retries, concurrent requests, snapshot reads, source verification and migration preservation | All 22 cases reviewed and passed in real isolated PostgreSQL/S3. No skipped storage cases. Four targeted profiling faults failed their intended assertions. |
| [DatasetProfile.test.tsx](../web/src/components/DatasetProfile.test.tsx) | Loading, explicit generation, duplicate prevention, dataset/column reads, exact text display, failures and stale responses | Eight rendered tests passed; API mocked. |
| [Browser journey](../web/e2e/ingestion.spec.ts), [Compose smoke](../docker/smoke.py) | Original upload/library journey plus saved profiling, independent column retrieval, refresh and service replacement | Shared with Phases 1/2, not copied journeys. See audit for the tested image. |

The final combined Phase 1-3 backend run passed **120 cases without skips**. All **66 frontend tests** passed; unreviewed frontend tests are not newly certified. Phase 1 and 2 affected fixtures/navigation/build checks were re-verified. CI includes the required storage baseline and negative controls; hosted CI execution remains unobserved.

Commands: `make test-profiling` for Phase 3 calculation/storage tests; `make test-storage` for the consolidated Phase 1-4 suite; `make test-ingestion-mutations` for the shared defect controls; `make container-smoke` for the real browser and persistent-volume journey. No additional storage harness or containers were introduced.

### Maintaining the register

Group future entries under their own numbered phase headings. Maintain this register alongside changes: add a location only after reviewing and running its relevant cases, link its evidence, and state its limits. A change to the tested behavior, assertions, fixtures or dependencies requires re-verification of the affected entry; until then, label it **re-verification required**. Mark a whole folder verified only when every test in it has been reviewed and its applicable checks executed. Keep historical evidence dated rather than presenting an old run as current verification.

### Phase 4: Organized dataset and column context

**Complete for its stated scope, verified 25 October 2026.** See [Phase 4 evidence](audits/phase4-context.md), including the authorized Gemini run and LangSmith readback.

| Reviewed location | Verified scope | Evidence and limits |
| --- | --- | --- |
| [memory/context/tests/](../causal_agent/memory/context/tests/) | Source references, all-column structure, bounded extraction/assessment, provider schemas, participant resolution, corrections and failure behavior | All 29 scripted cases passed. Model understanding has separate real-source evidence, not universal certification. |
| [server/tests/context/](../causal_agent/server/tests/context/) | Ownership, relational integrity, atomic publication, concurrent requests, saved reads and retained revisions | All 19 cases passed using the shared isolated PostgreSQL/S3 harness. |
| [DatasetContext.test.tsx](../web/src/components/DatasetContext.test.tsx) | Saved context panel, source rendering, relationship navigation, retries and stale responses | Seven rendered cases passed; model and API behavior are mocked. Two deliberate rendering defects failed the intended test. |

The final combined Phase 1-4 baseline passed **168 tests without skips**. All 19 backend and five frontend deliberate-defect controls passed; the five Phase 4 backend controls were repeated after the final extraction changes. All 74 frontend tests, the build, generated-contract reproducibility and relevant static checks passed. The rebuilt-image browser/container smoke passed both journeys and persistence across replacement of all four current services, re-verifying the affected Phase 1-3 storage/browser entries. The authorized Uruguay case passed in two Gemini calls; exact citations and all five column meanings were reviewed, and LangSmith readback showed no errors. Hosted CI remains unobserved. No other folder gains trust from these results.

### Consolidated Phase 4 execution

`make verify-phase4` orchestrates the existing owning-component tests, shared Phases 1-4 storage baseline, Phase 4 defect controls, frontend checks, generated-contract comparison and browser/container replacement. It produces one run-scoped report under `.artifacts/phase4/`; test files stay at their owning components. `make verify-phase4 WITH_MODEL=1` additionally runs the public, sourced model case and validates its trace. Model acceptance is explicitly separate in the report, and semantic review remains required. CI invokes the same runner with `ALL_CONTROLS=1` to retain all shared fault controls and never enables paid model calls. See the [Phase 4 verification map](audits/phase4-context.md#consolidated-command-and-policy-map) for commands, coverage and limits.

### Consolidated chat verification

`make verify-chat` keeps contract, repository, browser, container and deliberate-defect checks with their owning components and produces one report under `.artifacts/chat-verification/`. `make verify-chat WITH_MODEL=1` separately enables the recorded-source Gemini conversation and the three isolated lane boundaries. It never turns a green scripted run into semantic acceptance automatically. See the [26 October chat acceptance record](acceptance/consolidated-chat.md) for actual questions, corrections, Docker execution, citations, gate repairs, stopping behavior and deployment limits.
