# 03. Dataset profiling

[Component map](README.md) · [Source ownership](source-index.md) · [Testing](../testing.md)

## Purpose and boundary

Phase 3 adds read-only profiles to saved ingestion datasets. Upload remains independent: the person requests profiling from the dataset details page. Profiles describe original text, suggest representations and expose failures. They do not rewrite CSVs, confirm grain, create repaired versions or start conversations/models.

The separate analysis runtime retains its existing `Profile` contract and file cache. New owned profiles do not enter that cache or seed analysis memory. No compatibility loader connects these storage systems.

## Internal parts

| Owner | Responsibility |
|---|---|
| [inspection.py](../../causal_agent/profile/inspection.py) | Original-text measurements, conservative type suggestions, sentinel candidates and bounded key search |
| [report.py](../../causal_agent/profile/report.py) | Typed results, check outcomes, issues and owned API responses |
| [server/profiling](../../causal_agent/server/profiling/) | Ownership, verified object download, transaction/lock boundaries and PostgreSQL reads/writes |
| [DatasetProfile](../../web/src/components/DatasetProfile.tsx) | Dataset summary and independently fetched column details |
| [Existing profiler](../../causal_agent/profile/profiler.py), [data loader](../../causal_agent/profile/data.py), [cards](../../causal_agent/profile/pack.py) | Separate analysis contract, content-keyed disk cache and path-keyed process cache |

## Inputs, outputs and contracts

The new path accepts only an owned, ready original ingestion version. It streams its private S3 object into a temporary file, bounds and structurally validates it using ingestion rules, and verifies the stored SHA-256 and byte count before inspection. Temporary files and object streams close on success or failure.

CSV fields remain text. Empty fields differ from literal `NA`, `null`, `unknown`, whitespace and sentinel numbers. Suggested numeric, Boolean and ISO date types are not approved conversions. Ambiguous dates, leading zeros and unsupported numeric formats remain explicit limitations. Numeric summaries require every non-empty value to parse finitely; large integers outside exact float range do not receive summaries. Failed calculations retain visible failed checks.

Dataset facts include counts, duplicate parsed rows and candidate keys. Column facts include empty/whitespace counts, distinct values, eight top values, parsing counts, supported numerical summaries, suspected sentinels and format issues. Examples are limited to five per issue and 120 characters, with truncation indicated. Sentinel counts describe suspicion, not confirmed missingness. Key search covers all nonempty single columns and pairs among at most twelve non-single candidates, ordered by cardinality then position; no larger combinations. Keys do not establish row meaning.

Checks are `computed`, `unavailable` or `failed`. A report is `complete`, `partial` or `failed`; a warning alone does not make a report partial. Plain text columns need no numeric summary. See the authoritative typed contracts for all fields.

## Decision authority and dependencies

Code measures and suggests. This phase has no confirmation or repair operation. The person's original context remains verbatim in ingestion storage. The future repair agent can reference a profile, source checksum and stable column ID; no repair-agent API or context table is introduced here.

Computation depends on Python CSV/datetime, NumPy and existing sentinel constants. There is no new inference library. Server orchestration reuses the principal, private object client and PostgreSQL advisory-lock helper. It does not call the old profiler, desk, models or exploration.

## Storage and lifecycle

```text
User -> Dataset -> Original version
                   |-> stored-file record -> private CSV object
                   |-> original context
                   |-> profiling.columns (stable identities)
                   |-> profiling.profiles (dataset-level report and provenance)
                         |-> profiling.column_results -> stable column identity
```

The [migration](../../migrations/versions/0003_profiling.py) adds three tables without modifying ingestion records or objects. Composite foreign keys constrain owner and version across all result relationships. Column identity is scoped to a version, with unique position and normalized key, and survives retries/profiler changes. No column identity across future derived versions is implied.

A profile is keyed by version, source checksum, profiler version and settings fingerprint. Dataset and column results commit atomically. Complete matching profiles are reused; partial/failed profiles can be retried. Failed retries preserve the last useful report. Concurrent generation for one identity returns 409; interrupted processes release their lock and can be retried. There is no durable running job or automatic retry scheduler.

The dataset summary and column summaries are read in one repeatable-read snapshot. Detailed column retrieval selects only the requested result row. Stored reports are not copied to S3 or the filesystem; PostgreSQL JSONB holds bounded measurements while relationships remain relational. Existing app, database and object-store containers are reused. No deletion feature is added.

## User/API walkthrough

1. `GET /api/datasets/{id}/profile` loads the current saved report or `not_profiled`, without reading CSV bytes.
2. The person requests `POST /api/datasets/{id}/profile`; the page disables duplicate submissions and shows indeterminate progress.
3. The server verifies the original, computes results and commits them. Infrastructure failures return 503, not an empty report. Unknown/other-owner/unpublished datasets return 404.
4. The page shows counts, limitations and column summaries. Opening a column calls `GET /api/datasets/{id}/profiles/{profile_id}/columns/{column_id}`.
5. Refresh retrieves saved results. Source version and profiling timestamp remain visible. Original context and download remain available unchanged.

Responses use private/no-store metadata caching. Frontend route changes invalidate delayed responses. Profile and column retry errors remain visible; saved reports remain displayed. The development principal remains the only identity provider; this is not public multi-user deployment readiness.

## Verification

Tests live with their owners: `profile/tests/test_inspection_*`, `server/tests/profiling/`, and the profile component test. The existing browser journey and disposable Compose smoke cover the full vertical slice. The shared storage harness provisions a separate migrated database and private bucket for every integration test. See the [verification register](../testing.md) for observed results and remaining limitations.
