# Phase 4 verification record

**Complete for the stated scope, verified 25 October 2026.** Scripted tests, real storage, browser/persistence checks and the authorized Gemini/Vertex case passed. Hosted CI execution has not been observed.

Scope: automatic preparation of cited original study context, saved dataset and column views, shared relationships, PostgreSQL publication and ownership. No analysis or conversation integration.

## Evidence locations

- `causal_agent/memory/context/tests/`: bounded extraction, citation/identity validation, empty sources and correction behavior.
- `causal_agent/server/tests/context/`: real storage, HTTP ownership, source preservation, constraints, atomicity, concurrency and recovery.
- `web/src/components/DatasetContext.test.tsx`: saved reads, preparation, column/source views, retries and stale responses.
- `web/e2e/context.spec.ts`: real browser reads of published deterministic context fixtures.
- `.artifacts/ingestion-tests/`: baseline JUnit, deliberate-defect logs and result manifest.
- `.artifacts/context-acceptance/`: real Gemini outputs and trace identities, including failed attempts.

The shared harness provisions random disposable Compose services and a separate database/bucket for every test. It refuses required runs that select no integration cases or skip cases. Real model calls are excluded from ordinary tests and CI.

## Limits

Passing structural tests cannot establish that all future semantic interpretations are correct. The real-model evidence covers one retained study account. The updated shared harness and current-image smoke re-verify the affected Phase 1–3 storage and browser paths, not every application component.

## Reviewed and executed, 25 October 2026

| Scope | Observed evidence |
| --- | --- |
| `memory/context/tests/` | All 29 cases reviewed and passed: empty sources, all-column coverage, exact Unicode citations, ambiguous same-line quotations, report bounds, provider schemas with local validation, derived participant kinds, invalid references, material corrections versus notes, exhausted budgets and provider failures. These scripted cases verify control flow, not model understanding. |
| `server/tests/context/` | All 19 cases reviewed and passed against isolated PostgreSQL/S3: owned routes, immutable source retrieval, independent column reads, owner/version/source foreign keys, concurrent preparation, rollback/retry, revisions, migration preservation and publication rejection. |
| Combined Phases 1-4 | 168 passed, zero failures, errors or skips in the final baseline. Earlier ingestion/library/profiling tests were reused, not copied. The shared harness rejected empty required selections and deliberately skipped required cases. |
| `DatasetContext.test.tsx` | Seven rendered cases reviewed and passed. They cover saved reads, explicit generation, duplicate prevention, error/retry, plain-text source display, relationship navigation and stale dataset/column responses. APIs are mocked. |
| Backend fault controls | All 19 targeted defects detected by their named assertions earlier in this verification. After the final extraction changes, the full baseline and all five Phase 4 controls were repeated successfully: owner filtering, pack/version binding, citations, relationship participants and unresolved assessments. No setup/teardown errors or skips counted as successes. Manifests: `.artifacts/ingestion-tests/all-controls-before-live-fixes.json` and `phase4-final-controls.json`. |
| Frontend fault controls | Three ingestion adapter defects and two context presentation defects were detected. Each produced the intended test failure. Context controls deliberately break Unicode offsets and insert unsafe HTML rendering, using scratch copies only. |
| Static and contracts | Scoped Ruff and formatting passed, mypy passed for 253 source files, and all 12 import boundaries passed. Frontend ESLint and Prettier passed. OpenAPI and generated TypeScript reproduce identical hashes. All 74 frontend tests and the production build passed. Five trace-policy tests also passed. |
| Browser and container persistence | The final rebuilt image passed both browser journeys and replacement of app, sandbox, PostgreSQL and object storage. Saved metadata, exact original bytes, profiles, context summaries, column context and source detail survived replacement. Idempotent replay and cleanup retained the ready original. Disposable project cleanup completed. |

The extractor and assessor remain two roles in one bounded workflow. Corrections identify assertions and source lines. The shared citation gate rejects unresolved locations rather than allowing placeholder offsets to accidentally match source text. Participant kinds derive from the referenced identities. Both model roles share the provider-compatible schema implementation while local validation retains its full bounds. No second extraction path or model wrapper was added.

## Authorized real-model acceptance

The user explicitly authorized sending the Uruguay study account and five headers to Gemini/Vertex and recording them in LangSmith. CSV row values were not sent. The accepted run used `gemini-2.5-flash`, completed in two application model calls and needed no revision. Both validation gates passed. LangSmith readback contained 29 runs, two model calls and zero errors, with extraction, citation resolution, participant resolution, assessment and artifact spans under the same trace.

[Accepted LangSmith trace](https://smith.langchain.com/o/1c960ed1-6273-4ef3-a5fe-0b28fecc8a90/projects/p/f533b58d-5b58-408a-982d-5b6c31164a9f/r/01a17536-ac3b-71d1-b381-ae680faaa72d?poll=true). Local output, trace export and review are `.artifacts/context-acceptance/gov-transfers-acceptance.json`, `gov-transfers-acceptance-trace.json` and `gov-transfers-acceptance-review.json`.

Manual review covered all five columns: centered eligibility score, education, age, participation and support. The record preserved coding, measurement timing, eligibility/receipt rules, missingness qualifications and the sampling warning that counts by cutoff side do not establish manipulation. All quotations matched the unchanged source. Its five relationships represented assignment or derivation, without unsupported causal-effect claims. Source-supplied numerical facts remain attributed statements, not newly computed measurements. No causal graph, estimator or readiness decision was produced.

Original SHA-256 values remained unchanged:

- CSV: `c1f1699a0f9d2d3182aa4373c18830e0eefa5d32862b265fa5e642a9ad97d116`
- Study account: `8fe5f3f4d3430e2f91258a6876e5d8ee790d687a845dc58621196a8c0b4343fb`

Earlier attempts are retained as diagnostic evidence. They exposed provider schema restrictions, invalid relationship participants, over-strict presentation corrections and an unsupported causal relationship label. These led to shared schema handling, deterministic participant typing and more precise extraction/assessment instructions. The technically passing `gov-transfers-final-reviewed` attempt was rejected during manual review; only `gov-transfers-acceptance` is the final accepted output. Failed attempts and scripted retry tests demonstrate bounded stopping, rather than a claim that every model run succeeds.

## Docker resolution and remaining limits

The earlier build stalled retrieving `docker.io/docker/dockerfile:1`. Retrieval/build worked on this retry without changing daemon configuration or weakening isolation; the original transient cause remains unidentified. Final build `hnn4bmgbrefo7nmehuqrg854r` produced image `sha256:023a035d1478aaa6cbb1fd5c1bf2b1edde03a4c05d8901e47679ce30e848c342`. Project `causal-smoke-227a2b809291` completed replacement assertions and removed its disposable volumes. Log: `.artifacts/ingestion-tests/phase4-compose-final.log`.

The current checkout includes a sandbox worker from separate work. The smoke also passed its boundary, timeout, cancellation and fresh-task probes. Phase 4 adds no service and makes no claim to certify that separate agent. Browser context fixtures are deterministic and test display/persistence; Gemini semantic acceptance is the separate evidence above.

Hosted CI execution remains unobserved. Configured CI runs the shared storage baseline, targeted defects, browser/persistence smoke and static/contract/frontend checks. This record verifies Phase 4's organized original-context feature, not real authentication, later chat/repair/analysis integration or whole-application production readiness.

## Consolidated command and policy map

Run `make verify-phase4` from the repository root. It runs all repeatable Phase 4 code, storage, frontend, contract and deployment checks and prints the path to one `report.md`, with machine-readable `report.json` beside it. Each invocation uses a fresh `.artifacts/phase4/<UTC-time>-<id>/` folder. It retains logs, JUnit, defect-control manifests, browser failure traces and source hashes. Failed checks remain failed in the combined result; other checks still run. Missing dependencies, missing reports, empty selections and skipped required cases fail verification.

For the complete run including a new real-model case, use **`make verify-phase4 WITH_MODEL=1`**. This explicitly sends the retained public Uruguay study account and its five headers to configured Gemini/Vertex and LangSmith, never CSV rows. It verifies unchanged inputs, column coverage, exact quotations, the model-call budget, final gates and trace parentage, and retains the model output and full trace. Its semantic meaning and omissions still require review. An automated pass does not certify arbitrary future interpretations. The default command makes no model calls and marks that stage `not_requested`; it does not reuse an old acceptance as current evidence.

Prerequisites: locked Python/web dependencies (`uv sync --frozen`, `npm --prefix web ci`), Docker Engine/Desktop with Compose, and Chromium (`npm --prefix web exec -- playwright install chromium`). The live option also needs working Vertex credentials/project and LangSmith configuration. Tests use only the existing disposable PostgreSQL/S3 harness and smoke project; they do not connect to deployment storage.

| Requirement and boundary | Tests at their owner | Consolidated check |
| --- | --- | --- |
| Source fidelity, every column, bounded correction, unsupported assertions and provider failures | `memory/context/tests/test_extract.py`, including citation, ambiguity, correction and budget cases | Shared storage baseline |
| Owner/version/source identity and independent saved reads | `server/tests/context/test_api.py` roundtrip, owner isolation, version join and column read; `test_storage.py` column/source foreign keys | Shared storage baseline and five Phase 4 backend defect controls |
| Atomic writes, concurrency, retry and retained revisions | `server/tests/context/test_storage.py` interruption/retry, conflict and revision cases | Shared storage baseline |
| Context UI reads, source quotations, relationship links, retry and stale responses | `web/src/components/DatasetContext.test.tsx` | Frontend tests and two Phase 4 frontend defect controls |
| Real browser read/reload, database persistence and exact originals after process replacement | `web/e2e/context.spec.ts`, shared ingestion journey, `docker/smoke.py` | Browser and replacement check, all four services |
| Producer/consumer contracts, types, import boundaries and build | Current OpenAPI, generated TypeScript and static tools | Scratch generation comparison, lint/types/import checks and production build |
| Report credibility and actual model evidence | `scripts/tests/test_verify_phase4.py`, `evals/tests/test_context.py` | Runner/trace-policy tests; optional live model and trace check |

The Phase 4 test cases above were reviewed against setup, action, outcome, mocked dependencies and the defect they protect. Component tests mock the API and model deliberately; storage tests use real transactions and constraints; browser context fixtures prove rendering and persistence, not semantic extraction. Critical boundaries retain their existing deliberate-defect controls. No product tests were moved, copied or replaced by orchestration tests. Shared Phases 1-3 baseline tests run once through their existing owners.

CI uses the same entry point with `ALL_CONTROLS=1`, adding the existing Phase 1-3 defect controls without removing them or copying their implementation. CI remains model-free and uploads the run-scoped reports even on failure. Local passing evidence does not claim the hosted CI job has run.

### Consolidated runner verification, 25 October 2026

The complete command `make verify-phase4 WITH_MODEL=1` passed in run `20261025T234014Z-d78914ec`. Its single report is `.artifacts/phase4/20261025T234014Z-d78914ec/report.md`, with JSON, logs and evidence beside it. Results: 168 backend baseline cases without skips, five Phase 4 backend defects detected, two frontend defects detected, 74 frontend cases, 19 runner/evidence/trace-policy regressions, static/types/contracts/build checks and both browser journeys with container replacement. Test implementations remain at their owners.

The new real-model result completed in two calls. Trace `01a17573-f341-7c41-97de-2b75f43122a9` contains 29 completed spans and passing final gates. Exact citations, unchanged inputs, column coverage and bounded calls passed automated grading. Readback reviewed all five column meanings and the sampling qualification without unsupported causal-effect claims. The earlier aggregate run `20261025T233420Z-f8b3416b` remains failed because contradictory tracing flags prevented trace creation; the runner now sets both flags consistently, tests both modes and refuses trace-required acceptance before model invocation when tracing is disabled. This verifies the local runner, not hosted CI execution.
