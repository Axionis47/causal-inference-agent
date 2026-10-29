# Phase 3: Dataset and column profiling

Implementation and local verification, 23 October 2026.

## Implemented boundary

The existing dataset details page offers explicit profiling and saved dataset/column reports. PostgreSQL schema `profiling` has three tables: stable version-scoped column identities, profile provenance/dataset results, and independently retrievable column results. Composite foreign keys prevent cross-owner and cross-version references. The source remains the immutable ingestion version and private CSV object. Original context is not changed.

Parsing preserves raw field text. Numeric/date/Boolean suggestions are conservative and do not write conversions. Unsupported representations, parsing exceptions and failed checks remain visible. A sentinel candidate is not confirmed missingness. Candidate keys do not establish grain. Result examples and frequency lists are bounded. The separate analysis runtime and its old profile contract remain unchanged; no repair, context-editing, derived-version or model feature was added.

## Test ownership and observed checks

- `profile/tests/test_inspection_columns.py`: field preservation, sentinel candidates, conservative dtypes, numerical values, failure counts and bounded examples.
- `profile/tests/test_inspection_dataset.py`: BOM/quoting/newlines, duplicate rows, key search and determinism.
- `profile/tests/test_inspection_failures.py`: partial results and explicit failed checks.
- `server/tests/profiling/test_api.py`: real owned HTTP, read-only retrieval, independent columns, partial reports, unavailable storage and unpublished records.
- `server/tests/profiling/test_storage.py`: real composite FKs, stable IDs, atomic publication, concurrency, retries, checksums, consistent snapshots, migration preservation and interrupted streams.
- `DatasetProfile.test.tsx`: eight rendered UI cases with mocked API, including delayed initial/generation/column responses.
- Existing browser/Compose smoke: two actual browser uploads, library return, explicit profiling, column expansion, refreshed saved reports, exact originals and persistent dataset/column reports after replacement of all three containers.

The dedicated combined run passed 120 cases without skips: 36 profile cases (29 new), 48 Phase 1 cases, 14 Phase 2 cases and 22 Phase 3 storage/API cases. Evidence: `.artifacts/ingestion-tests/phase3-final.xml`. Every integration case uses the shared harness's own migrated database and private bucket. No deployment data or volumes were used.

All 66 frontend tests passed. Python Ruff/format, mypy (235 source files), 12 import contracts, five trace-policy tests and frontend lint/type checks/build passed. Ordinary server discovery passed 27 tests and explicitly skipped 66 storage cases; the dedicated run above, not those skips, establishes storage coverage.

Four profiling negative controls removed owner filtering, removed the dataset-version restriction on column reads, accepted a numeric summary after discarding an invalid value, and marked a failed check complete. Each failed its named assertion without setup/teardown errors. The shared runner checks required-selection failures and retains JUnit/log evidence under `.artifacts/ingestion-tests/`. All 14 backend controls, including the ten existing ingestion/library controls, detected their intended defects without setup errors. Empty required selection and forced skips were correctly rejected. The three existing frontend upload negative controls also passed their credibility checks.

The final Docker image `sha256:9ce6dff7c74b4cfa7ed570a86d60a78db59f920808111357603bd39f7110a233` passed the real Chromium journey and all-service replacement smoke on Docker Desktop Linux/arm64. All three container IDs changed, and both dataset and individual column profile responses remained identical. The disposable project and its volumes were removed afterward.

OpenAPI and generated TypeScript freshness passed by comparing hashes before and after regeneration. Changed documentation links and `git diff --check` passed. No application-data migration or cleanup was run against the user installation; schema migration was verified in disposable test databases and is applied by the existing startup initialization on deployment.

## Accepted follow-up limitations

On 23 October 2026, follow-up column probes found two unresolved precision defects:

- `1e-999` and `2e-999` produced a complete numeric profile with zero-valued summaries through floating-point underflow.
- `9007199254740993.0` produced a complete numeric profile with the rounded value `9007199254740992.0` and no precision warning.

These affect calculated summaries, not original CSV bytes. The existing passing tests did not cover these cases. Maximum-width (200-column) performance also remains unverified.

After these limitations were explained, the user accepted them for this milestone. **Phase 3 is complete within the accepted scope, with these defects deferred rather than fixed.** This documentation update adds no runtime fix, regression test or broader numerical correctness claim. Repairs, saved conversions and analysis integration remain outside the milestone.

## Limits

- Development principal only. This is not a production authentication or public hosting claim.
- No new inference library, general locale conversion, semantic grain inference, panel diagnostics or repair execution.
- Synchronous generation is bounded by configured ingestion file/row/column/cell limits; peak-load performance was not certified.
- Retrying a partial profile recomputes the report. Infrastructure errors preserve the previous committed report and return a service error. Process termination does not create a durable job; a later request retries.
- Container replacement proves persistent-volume behavior, not backup restoration.
- Hosted CI has been configured but not observed. Full causal/model acceptance is outside this feature and no model was called.
