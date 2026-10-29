# Phase 2: Return to your datasets

## Cleanup inventory, 23 October 2026

Resolved locations were checked before deletion:

| Location | Observation and action |
|---|---|
| `data/web/` | Empty; no web-created dataset identities or artifact references to follow. |
| `.artifacts/web/checkpoints.sqlite` | Zero bytes. No open file handle or active application server found by process/file-handle inspection. Removed this empty checkpoint. |
| `.artifacts/web/uploads/` | Empty; no staged objects to remove. |
| `.artifacts/runs/` and `data/memory/` | No association could be established through old web records, so retained. Dataset memory includes fixture/study inputs and is not blanket-deleted. |

Original CSVs, the evaluation registry, supplied study inputs, fixtures and PostgreSQL/S3 ingestion data were preserved. No application writers required stopping. Cleanup was a one-time workspace action, not startup behavior. It does not claim to have cleaned deployments or external volumes.

## Verification

Verified locally on 23 October 2026:

- Phase 1 and Phase 2 real-storage suite: **62 passed**, no skips (48 ingestion and 14 library cases). Each storage case uses a fresh migrated database and private bucket in harness-owned services.
- Backend negative controls: **10 of 10 detected**, including all three new library faults: removed owner filter, inclusive pagination boundary causing repeated results, and removed ready-status filter. Each named assertion failed without setup/teardown errors. A passing baseline was required first.
- Frontend source negative controls: **3 of 3 detected**. Full frontend suite: **58 passed**, including three upload form, three library/shared-state and three detail-view cases.
- Ordinary server suite: **27 passed, 41 explicitly skipped** before the final three malformed-cursor cases were added. The dedicated 62-case run above exercised all required real-storage cases, including those new cases.
- Real Chromium journey passed: upload two datasets, return through the library, reopen each, refresh and download exact CSV bytes and original context.
- Fresh-image Compose smoke passed: all three service container IDs changed, and the owned list, metadata, idempotent replay and bytes survived. Final tested image: `sha256:3f2a334a8dcc8632dfe90cbe00ad29588ff1a2b2b41189a435ce654763128842`.
- Ruff, mypy, frontend lint/types/build, generated-contract freshness and all 12 import contracts passed. One upstream Starlette/AnyIO deprecation warning remains.

Machine-readable local evidence: `.artifacts/ingestion-tests/phase2-final.xml`, `results.json`, `web-results.json` and per-control JUnit/logs. The existing CI storage job now runs both phases and their negative controls, plus the browser/persistence smoke. Hosted CI execution has not been observed.

The old library-only tests and assertions were removed with the corresponding API. Session tests continue through their own fixture and session API; session-manager checkpoint cleanup remains tested directly. No model judgments, prompts or causal decision logic changed.

Application/schema imports may recreate an empty checkpoint file for the separate session runtime. After verification it was checked and removed again; no old conversation records were retained there.

## Phase 2 verification map

| Responsibility | Tests / enforcement |
|---|---|
| Owned list and metadata shape | `server/tests/library/test_library.py::test_owner_filter` and `test_http_read_contract` |
| Stable pagination and ready-only visibility | `test_pagination`, `test_only_ready`, with independent negative controls |
| Malformed limits/cursors and other-owner cursors | `test_invalid_query`, `test_other_owner_cursor`, `test_cursor_field_types` |
| Failure is not an empty library; reads do not fetch CSVs or alter rows | `test_unavailable_is_not_empty`, `test_http_read_contract` |
| Shared collection, retry state and stale responses | `web/src/pages/Landing.test.tsx` |
| Exact plain-text context, download links and detail navigation | `web/src/pages/DatasetDetails.test.tsx` |
| Upload navigation and retained retry behavior | `web/src/pages/NewDataset.test.tsx` |
| Complete return journey and deployment persistence | `web/e2e/ingestion.spec.ts`, `docker/smoke.py` |

## Limits

Development identity is still a placeholder. This is not production authentication or a release approval for the broader application. Profiling, chats, analysis, search, editable context and deletion of newly ingested datasets remain outside Phase 2. Source fixtures and unassociated runtime artifacts were retained, not migrated or exposed through the library.
