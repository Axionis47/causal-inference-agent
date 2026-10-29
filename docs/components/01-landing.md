# 01. Landing and dataset library

[Component map](README.md) · [Testing policy](../testing.md) · [Phase 2 evidence](../audits/phase2-library.md)

## Purpose and boundary

Phase 2 lets the current user return to saved ingestion datasets. `/` lists owned datasets; `/datasets/:datasetId` opens the original metadata and study context, with a CSV download link. It does not start profiling, conversation or analysis. Real login remains deferred; development uses the server-side principal and production rejects that placeholder.

## Internal parts

| Part | Responsibility |
|---|---|
| `server/ingestion/library.py` | Read-only PostgreSQL query, owner filtering and cursor pagination |
| `server/ingestion/routes.py` | Principal, input validation, generated response contracts and errors |
| `web/src/hooks/useDatasetLibrary.ts` | One shared list state, pagination, refresh and stale-response protection |
| `Shell`, `Sidebar`, `Landing`, `DatasetCard` | Render that shared collection and navigation |
| `DatasetDetails` | Retrieve the selected dataset automatically; render original context and download link |

## Inputs and outputs

`GET /api/datasets?limit=20&cursor=...` returns `DatasetList` with `items` and `next_cursor`. Each `DatasetListItem` contains dataset/version IDs, title, filename, byte size, upload date, saved status and context presence. Full context and CSV contents are excluded from the list. Exhaustive fields are authoritative in [server models](../../causal_agent/server/models.py) and generated [OpenAPI](../../web/openapi.json).

## Contracts

- The server derives the owner; all list, detail and download queries enforce ownership.
- Only ready ingestion operations appear. The list does not consult the fixture registry, desk, profile cache or CSV objects.
- Limit defaults to 20 and accepts 1 through 100. Ordering is descending creation timestamp, then descending ID. The cursor encodes owner and the last ordering values; it is a position, not a credential. Mismatched-owner or malformed cursors return 422. Every query independently filters by owner.
- Fetching one extra record establishes whether a next page exists. Refresh starts over; pagination is not a historical snapshot.
- Metadata uses `Cache-Control: private, no-store`. Storage failure returns 503; it never becomes an empty collection. Missing or other-owner detail/download returns 404.
- A successful upload navigates to the canonical details URL and refreshes the shared collection. There is no old confirmation-query compatibility route or name-based deletion API.

## Decision authority

The person chooses a dataset or requests more results. Code validates identity, filters and sorts metadata, and prevents delayed responses from overwriting newer requests. No model participates.

## Dependencies

The library uses the existing ingestion store, principal and relational schema. A second Alembic migration indexes `(owner_id, created_at DESC, id DESC)`. It preserves existing records. React consumes generated API types; sidebar and homepage do not maintain separate copies of the collection.

## State and storage

PostgreSQL owns dataset metadata; private S3 holds exact original CSVs. Browsing only reads metadata. The frontend holds loaded items, cursor, error and loading state in memory. A failed next-page request keeps existing items and offers retry. Sidebar layout preference may persist locally; dataset contents are not stored there.

## Worked walkthrough

Upload through `/new` → receive durable confirmation → navigate to `/datasets/:datasetId` → read original context or download CSV → return to `/` → select either of several saved datasets → refresh the details URL and retrieve it again. An empty library prompts upload; an unavailable library shows an error. Older conversations are not shown in this library.

## Verification

[Backend Phase 2 tests](../../causal_agent/server/tests/library/test_library.py) exercise owned queries with real storage. Frontend tests beside `Landing` and `DatasetDetails` cover rendering, pagination, errors and stale responses. [Browser journey](../../web/e2e/ingestion.spec.ts) connects two uploads to the library and downloads. [Compose smoke](../../docker/smoke.py) checks listing and retrieval after service replacement. See the evidence report for observed results and limitations; passing this phase does not certify production authentication or later analysis phases.
