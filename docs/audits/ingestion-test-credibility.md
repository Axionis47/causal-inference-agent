# Ingestion test credibility audit

Audited 22 October 2026. Scope: every ingestion Python test case (16 collected cases), both rendered upload tests, the Docker smoke script, and their configuration/CI. Other application tests were not individually audited. Application and test implementations were not changed during this audit.

The findings below are the historical pre-repair audit. See the [repair evidence](#repair-evidence-23-october-2026) for the current harness.

## Verdict and evidence

This is useful initial coverage, but not a sufficient acceptance harness. Passing counts overstated the evidence for cleanup, ownership constraints, concurrency and absence of model calls.

A fresh disposable PostgreSQL/SeaweedFS deployment ran all 16 Python cases successfully. Both frontend tests were rerun. Two targeted mutation experiments then demonstrated false passes:

1. Replaced `Store.cleanup` with `lambda self: 0` in a temporary pytest plugin. `test_real_failed_write_retry_cleanup_and_publication` still passed.
2. Dropped only `ingestion.contexts`' composite `(version_id, owner_id)` foreign key in the disposable database. `test_real_database_constraints_and_immutable_context` still passed.

These mutations did not touch repository implementation or production data. They were applied after the baseline and the disposable services were removed afterward. The smoke script was inspected, not rerun during this audit. Its previous successful execution remains separate evidence.

## Findings requiring repair

### P1: Wrong-owner insertion can fail for the wrong reason

[Ownership test](../../causal_agent/server/tests/test_ingestion.py#L124) first creates a version that already has a context, then inserts another context against that version with a different owner. `contexts.version_id` is unique. The broad `IntegrityError` assertion therefore accepts the uniqueness violation even if the composite ownership constraint is gone. The deliberate foreign-key removal experiment confirmed this.

Repair: create the target version without a context; ensure the other owner exists; attempt the cross-owner context insertion. Assert PostgreSQL foreign-key violation SQLSTATE `23503` and the intended constraint, then independently verify no illegal record was committed. Removing that foreign key must make the test fail. Test the other ownership edges separately rather than inferring all constraints from this one.

### P1: Cleanup can be entirely disabled without failing its test

[Failure/recovery test](../../causal_agent/server/tests/test_ingestion.py#L94) verifies no dataset was published and a subsequent retry succeeds. It never checks whether the orphan object disappears between cleanup and retry. Retry writes the object again. The cleanup-no-op mutation confirmed the false pass.

Repair: prove the orphan exists before cleanup, then assert an S3 not-found response afterward. Assert the retained operation state. Call cleanup again to establish repeatability. Test recent pending objects, aged ready objects, active writer locks and object-deletion failure independently. Close the streamed download body with `finally`.

### P1: CI does not require the real storage tests

The [storage fixture](../../causal_agent/server/tests/test_ingestion.py#L54) skips when `INGESTION_TEST_DATABASE_URL` is absent. [CI](../../.github/workflows/ci.yml) supplies neither that variable nor PostgreSQL/S3 services, and does not run the container smoke script. Its normal green result can omit all six integration tests.

Repair: add a dedicated integration job that provisions disposable services, applies the actual migration and runs marked integration cases. Missing configuration or zero collected integration cases must fail that job. Keep lightweight unit runs independently available.

### P1: Test storage isolation is a convention, not an enforced boundary

The fixture changes only the database URL. Object-store endpoint and bucket come from ordinary application configuration, with no unique test bucket or disposable-resource verification. The cleanup test invokes a global cleanup sweep, which can delete aged unpublished objects unrelated to the current test in the configured database/store. Random user IDs prevent some collisions but do not isolate that sweep. Tests also accumulate records and objects because fixture teardown only disposes the engine.

Repair: the harness must own a dedicated database and bucket/prefix for the run, initialize them, and clean only those resources. Fail fast if that isolation cannot be established. Never point this harness at an ordinary application deployment. This audit used dedicated Compose volumes, not application storage.

### P2: Concurrency coverage is only a lock-conflict check

[Concurrency test](../../causal_agent/server/tests/test_ingestion.py#L116) manually acquires the application's lock, calls `ingest` synchronously and expects a conflict. This genuinely tests PostgreSQL lock contention between connections. It does not exercise two submissions competing through validation, operation creation, object writing and publication.

Repair: run independent clients/workers with controlled synchronization at the write boundary. Assert one published dataset/version/file, safe conflict or replay responses, no duplicate object, eventual retrieval, and correct different-payload conflict behavior. Include a writer exiting while holding the operation lock.

### P2: HTTP and frontend mocks leave the transport seam untested

The [HTTP test](../../causal_agent/server/tests/test_ingestion.py#L136) builds a new minimal FastAPI app and overrides identity and storage. The [frontend tests](../../web/src/pages/NewDataset.test.tsx) mock the entire API module. Neither exercises the real XMLHttpRequest adapter against the full application. Field names, omitted context/file, error parsing, request wiring and app middleware registration can regress between these layers.

Repair: test the real API adapter's multipart fields, exact payload, progress, timeout and non-JSON/error behavior. Add a real browser journey against the packaged app, including reopening the confirmation URL. Use the actual app factory for server wiring tests. Identity tests must exercise actual development/production configuration without replacing the identity dependency.

### P2: Several claims exceed what their assertions establish

- “No runtime dependency” is inferred from mounting one router, without explicit guards against profiler/desk/model entrypoints.
- The smoke's empty library assertion does not prove no profiler or model was invoked.
- Recreating a `Store` proves a new connection/client can read persistence; it is not process-death recovery.
- The smoke uses `--no-build` against a mutable local tag. It does not record/assert which checkout built the image.
- The smoke fetches HTML with urllib, so it does not test whether the React application renders or its upload adapter works.
- The smoke requests service recreation but does not compare container identities to establish which services were replaced.

Repair the assertions or narrow the test names and documentation. Add failing guards for forbidden runtime calls and identify the tested image/commit. Do not count a manual green run as continuous enforcement.

## Individual Python case review

All source references in this table point into [test_ingestion.py](../../causal_agent/server/tests/test_ingestion.py).

| # | Case | Credible evidence | Limit / required improvement |
|---|---|---|---|
| 1 | `test_invalid_csv`: empty bytes | Real validator rejects empty input with `InvalidUpload`. | Valid narrow unit case; assert an actionable error reason and separate HTTP non-publication coverage. |
| 2 | Header-only CSV | Real validator rejects a table without data records. | Valid narrow unit case; does not prove no records/objects persist through HTTP. |
| 3 | Duplicate raw headers | Rejects two identical headers. | Valid narrow unit case; add distinct valid-header counterpart and identify the header error. |
| 4 | Normalized header collision | Rejects `exam score` versus `exam-score`. | Valid case; assert normalization-specific reason, plus non-colliding punctuation and Unicode cases. |
| 5 | Short row | Rejects fewer fields than the header. | Valid case; also test extra fields, later malformed records and quoted embedded newlines. |
| 6 | Unclosed quote | Rejects an unterminated quoted field. | Valid case; add valid escaped-quote counterpart. |
| 7 | Invalid UTF-8 | Rejects invalid encoding. | Valid case; include valid multibyte text. Current positive fixture covers BOM, not multilingual text. |
| 8 | NUL byte | Rejects binary NUL in the CSV. | Valid case; assert useful error classification rather than any rejection. |
| 9 | `test_exact_bytes_and_limits` | Positive BOM/CRLF/quoted-comma/multiline fixture passes; stream is unchanged and rewound. Each of four very small limits rejects it. | Partial: SHA-256 return value is never checked. No exactly-at-limit versus one-over boundaries, no counting-stream check of bounded reads. Split limits into individually named parameterized cases. |
| 10 | `test_request_bound_and_production_identity` | Tiny request cap produces 413; direct principal invocation refuses production mode. | Partial: unrelated concerns combined; request lacks required title/key, although middleware rejects first. No full app wiring, multi-chunk body, disconnect, or HTTP production identity test. Catch `HTTPException`, not generic `Exception`. |
| 11 | `test_real_roundtrip_isolation_and_idempotency` | Real database and S3 roundtrip; exact context/bytes; service-level cross-owner get/download rejection; unchanged replay response; changed context conflict; different user gets a different dataset; fresh client can read. | Useful integration case. Missing row/object counts, changed CSV/title/filename conflict cases and server-derived identity wiring. Fresh-client construction is not a crash test. |
| 12 | `test_real_failed_write_retry_cleanup_and_publication` | Controlled post-write exception leaves no dataset, and retry succeeds. | **Reject as cleanup evidence:** confirmed no-op-cleanup false pass. No commit-acknowledgement loss, checksum mismatch, interrupted process, or deletion failure coverage. |
| 13 | `test_real_concurrent_submission_fence` | Real lock held on one connection causes another ingestion call to conflict; later call succeeds. | Keep as a lock test; **not sufficient concurrency acceptance**. No simultaneous ingestion calls or persisted cardinality assertions. |
| 14 | `test_real_database_constraints_and_immutable_context` | Context update is rejected by some database error. | **Reject ownership claim:** confirmed foreign-key-removal false pass. Assert intended FK and trigger errors, then unchanged data. Original-file/version update triggers are not exercised. |
| 15 | `test_real_http_ingestion_has_no_runtime_dependency` | Real router handles multipart input, returns stored metadata/file and denies another supplied principal's metadata read. | Useful route integration with explicit overrides, but not full app/auth wiring. No runtime-call guard, unauthorized HTTP file request, missing context submission, malformed request or HTTP error/retry checks. |
| 16 | `test_real_private_objects_refuse_anonymous_reads` | Actual unsigned S3 GET of an existing stored object receives 403. The successful ingest verified the object beforehand. | Credible narrowly scoped privacy test. Does not establish public deployment security or isolate test storage. Close the anonymous SDK client. |

## Individual frontend case review

Source: [NewDataset.test.tsx](../../web/src/pages/NewDataset.test.tsx). Vitest currently includes both `.test.ts` and `.test.tsx`, so these tests are discovered.

| Case | Credible evidence | Limit / required improvement |
|---|---|---|
| Preserve form after failure, reuse key, confirm | Real React rendering and handlers keep entered title/context after a mocked failure; two calls use equal keys; mock success displays confirmation/download link. | API is fully mocked. Assert all arguments, actual selected file bytes and a nonempty key. Changing the transmitted context to an empty string could escape these assertions. Use realistic UUID/checksum fixtures; verify changed input receives a new key; test confirmation reload and retrieval errors. It does not establish “no chat” execution on the backend. |
| Disable repeat submission and display progress | A pending mocked operation sets 40% display, disables the button and prevents another button click from calling the mock. | Credible narrow component test. Does not test real upload progress, completion/waiting transition, keyboard/form races, failed-operation re-enabling, aborts or timeout. The deliberately unresolved promise is suitable for a pending state, not a transport test. |

## Container smoke review

Source: [docker/smoke.py](../../docker/smoke.py).

Credible existing evidence: fresh Compose project/volumes, real multipart request, actual read-back bytes/context, identical replay response and persisted retrieval after the recreation command. Its random project name and `finally` cleanup are materially stronger isolation than the integration fixture.

Limitations: it is a happy-path HTTP smoke, not browser E2E, concurrency, process-kill recovery or a comprehensive immutability test. The cleanup check uses only a ready object and creates no eligible orphan; cleanup could be a no-op. A stale local image can be tested because `--no-build` is unconditional. There are no explicit forbidden-call guards or container-ID comparisons. Secret-bearing parent environment values can still affect unoverridden Compose variables despite `--env-file /dev/null`; the script explicitly clears only LangSmith tracing/key. No model call is expected from the inspected implementation, but the smoke assertion does not establish that property.

## Missing acceptance cases

1. Cleanup-disabled and ownership-FK-disabled mutations must fail after repairs.
2. Exact boundary acceptance/rejection for bytes, rows, columns and cells; blank headers, extension validation, extra-width rows and large fields.
3. Real concurrent duplicate/different submissions, connection failure before publication, lost commit acknowledgement and actual worker death.
4. Read-back checksum mismatch must prevent publication.
5. Aged ready files survive cleanup; aged failed/pending objects disappear; recent/locked objects survive; failed deletion can be retried.
6. User spoofing through form/header input fails to change ownership; metadata and downloads both enforce actual identity wiring.
7. File/version/context immutability checks fail on their intended constraints and reread unchanged data.
8. Real XHR adapter and a browser submission/reload/download flow against the built app.
9. A required disposable integration CI job with migration setup, no silent skips and explicit artifact/image identity.

Repair order: isolate the harness first; fix the two confirmed false passes; require integration in CI; then add concurrency/failure and browser seam coverage. Preserve the valid unit checks instead of replacing everything with mocks or a single giant journey test.

## Repair evidence, 23 October 2026

The ingestion harness was repaired after the audit above. Scope remains ingestion; this is not an audit of every test in the application.

### Isolation and continuous enforcement

[The storage harness](../../causal_agent/server/tests/ingestion_resources.py) creates a random Compose project with loopback-only PostgreSQL/S3 ports and sanitizes environment interpolation. Each test receives a new PostgreSQL database, applies the real Alembic migration, and creates a separate private bucket. Teardown deletes only those generated resources. It does not accept deployment storage addresses. The mutation baseline also ran with deliberately unusable deployment database and S3 environment values, establishing that those values were not used.

`make test-ingestion` requires the real tests. Ordinary pytest explicitly skips the Docker group. The required mode fails if none are selected or any are skipped; the mutation runner independently checks both failure conditions. [CI](../../.github/workflows/ci.yml) now has a dedicated ingestion job that runs the baseline, deliberate defects, fresh-image browser journey and persistence smoke, then uploads evidence even on failure. This job is configured; a hosted GitHub Actions execution was not observed during this repair.

### Case repairs

| Original finding | Replacement evidence |
|---|---|
| Validator could reject for an unrelated reason | Individually named invalid inputs assert actionable reasons; positive BOM/Unicode/quoted-newline/large-field counterparts assert exact bytes, SHA-256 and bounded reads. Each of four limits tests exact acceptance and one-over rejection. |
| Cleanup no-op falsely passed | Create and confirm an eligible orphan first, then assert actual S3 absence, retry publication and record counts. Separate tests retain recent, locked and ready objects and recover from deletion failure. |
| UNIQUE violation masked missing ownership FK | Create a version without context, insert the wrong owner, and require SQLSTATE 23503 plus the exact composite FK name. Four other ownership edges have independent constraint assertions. |
| Broad immutability exception | File, version and context cases require the intended trigger exception and unchanged stored data. |
| Manual lock was mistaken for concurrent submission | Two real ingestion calls overlap at object writing. Assert the second conflicts, replay returns the first result, and only one file/version/dataset is published. |
| Incomplete failure recovery | Corrupt stored bytes before verification; lose commit acknowledgement after a real commit; terminate a spawned writer after object storage but before publication. Assert publication/retry state and exact bytes. |
| Router-only identity/runtime evidence | Full application tests use the actual development principal wiring and real storage. A spoofed form owner gains no access; metadata and downloads deny another principal; production refuses development identity. Explicit profiler/session/model guards are installed. |
| Weak HTTP rejection coverage | Invalid CSV, oversized upload and malformed/non-string/NUL/oversized context leave no records or objects. Omitted context succeeds without creating a context record. |
| Fully mocked frontend API | Separate actual-XHR-adapter tests inspect multipart fields and bytes, progress and HTTP/network/timeout failures. Rendered form tests cover retry state, keys, busy state and confirmation retrieval. |
| HTTP-only smoke and stale image | Build the checkout, print its image digest, use real Chromium controls to submit/reload/download, then compare all service container IDs after explicit recreation and reread exact persisted data. |

### Deliberate defects detected

[Backend controls](../../scripts/ingestion_mutations.py) apply runtime replacements or DDL only inside disposable test resources. Each requires exactly the named test to fail with no setup/teardown errors or skips. A passing baseline is required first. [Frontend controls](../../scripts/ingestion_web_mutations.py) modify the actual adapter source in temporary copies and require its named transmission test to fail.

| Broken behavior | Test that failed |
|---|---|
| Cleanup does nothing | `test_cleanup_removes_orphan_and_retry_publishes_once` |
| Remove context ownership FK | `test_context_ownership_fk_rejects_wrong_owner` |
| Remove context immutability trigger | `test_original_records_are_immutable[context]` |
| Disable ingestion advisory lock | `test_two_submissions_compete_through_publication` |
| Skip object read-back verification | `test_corrupted_readback_cannot_publish` |
| Bypass owner filtering on reads | `test_roundtrip_owner_isolation_and_replay` |
| Invoke profiling during ingestion | `test_full_app_roundtrip_identity_and_no_runtime_calls` |
| Drop context from multipart payload | Actual adapter transmission test |
| Send to the wrong upload route | Actual adapter transmission test |
| Replace the submission key with a constant | Actual adapter transmission test |

All ten deliberate defects were detected. These targeted controls establish sensitivity to those faults, not exhaustive mutation coverage or proof that no defects remain.

### Actual defects exposed while repairing the harness

The browser test initially failed because multipart serialization converted LF to CRLF in supplied context. The API now accepts JSON-encoded context text and decodes it exactly once, and the frontend sends `JSON.stringify(context)`. Generated OpenAPI and TypeScript were regenerated. The exact-text browser assertion now passes; no plaintext compatibility branch was added.

The container-ID assertion also initially failed because the recreation command replaced only the app. The smoke now explicitly recreates PostgreSQL, object storage and the app and verifies all IDs changed before reading persisted data.

### Observed local results

- Final ingestion suite: **48 passed**, including all real storage cases, no skips.
- Backend deliberate-defect runner: baseline passed, **7 of 7** named faults detected without harness errors.
- Frontend adapter baseline and **3 of 3** source mutations verified; complete Vitest suite: **54 passed**.
- Real Chromium upload/reload/download passed. Fresh-image Compose smoke passed after explicit replacement of all three containers and exact CSV/context retrieval. Tested image: `sha256:314711e3917b2e140a3186cb028d0c8e3fc6a00132c3f9f773d9e28493ce61fb`.
- Ordinary server suite: **28 passed, 30 explicitly skipped** (the dedicated 48-case run above exercised the storage cases).
- Server Ruff checks and formatting, frontend lint/type checks and mypy passed. One upstream Starlette/AnyIO deprecation warning remains.

Local machine-readable evidence lives in `.artifacts/ingestion-tests/`: `final-baseline.xml`, per-mutation JUnit/logs, `results.json` and `web-results.json`. Browser failure traces are retained by Playwright; CI uploads its test-results directory. None of these checks call a model or establish production authentication, later profiling/conversation integration, load capacity or backup restoration.
