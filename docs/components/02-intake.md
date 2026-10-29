# 02. Dataset intake

[Component map](README.md) · [Terms explained](terms.md) · [Source ownership](source-index.md)

## Purpose and boundary

Save one CSV, its title and optional original study context under a server-derived user. This milestone ends at durable ingestion and confirmation. It does not profile, create a conversation, update causal memory or call a model. Phase 2 lists these saved datasets and opens their original context. Conversation and analysis integration remains deferred.

## Internal parts

| Part | Implementation | Responsibility |
|---|---|---|
| Form | [NewDataset](../../web/src/pages/NewDataset.tsx), [UploadDrop](../../web/src/components/UploadDrop.tsx) | Collect input, show progress/errors and saved confirmation |
| HTTP boundary | [routes](../../causal_agent/server/ingestion/routes.py) | Body limits, development principal, typed responses, owned downloads |
| Structural validation | [validation](../../causal_agent/server/ingestion/validation.py) | UTF-8 CSV structure and bounded dimensions, without profiling |
| Publication/recovery | [service](../../causal_agent/server/ingestion/service.py) | Idempotency, PostgreSQL locks, publication and cleanup |
| File storage | [objects](../../causal_agent/server/ingestion/objects.py) | Private S3 writes and byte verification |
| Relational constraints | [schema](../../causal_agent/server/ingestion/schema.py), [migration](../../migrations/versions/0001_ingestion.py) | Ownership, identities and immutable originals |

## Inputs and outputs

`POST /api/datasets` accepts multipart `title`, `file`, optional `context`, and `idempotency_key`. It returns [IngestedDataset](../../causal_agent/server/models.py), including dataset/version IDs, filename, checksum, size, exact context and saved status. `GET /api/datasets/{dataset_id}` retrieves that record. `GET /api/datasets/{dataset_id}/file` streams the authorized original. The multipart `context` field is a JSON-encoded string (the browser sends `JSON.stringify(text)`), decoded once at the API boundary. This prevents browser multipart serialization from changing line endings in the supplied account. Omission means no account; invalid JSON, non-string values, NULs and decoded text over 100,000 characters are rejected. There is no plaintext compatibility path. Exhaustive fields are in generated [OpenAPI](../../web/openapi.json).

## Contracts

| Boundary | Producer / consumer | Enforcement and failure |
|---|---|---|
| Form → API | Browser / ingestion routes | Required title/key/file; 422 validation, 413 oversized request |
| CSV → validator | Upload spool / validation | UTF-8, unique normalized headers, consistent record widths and configured limits; 422 without publication |
| API → storage | Routes / service | Owner derived server-side; unavailable configuration/storage returns 503 |
| Retry → existing operation | Browser / PostgreSQL | Owner/key unique; changed content or in-flight write returns 409 |
| Objects → publication | S3 / service | Read-back checksum and size before a single relational publication transaction |
| Retrieval → owner | PostgreSQL / API | Dataset belongs to principal, otherwise 404; no raw storage keys exposed |

## Decision authority

The person supplies the title, CSV and original account. Code validates structure and persistence. The account is not interpreted or promoted to a causal claim. There is no initial-context editing endpoint; database triggers reject updates to original file, version and context records.

## Dependencies

The owner is `server/ingestion/`. It depends on common configuration and column-key normalization, SQLAlchemy/Psycopg, Alembic and an S3 SDK. It does not import profiling, desk, memory, lane or model code. Generated TypeScript contracts connect the frontend to the routes.

## State and storage

PostgreSQL schema `ingestion` holds users, operations, datasets, original versions, file metadata and optional context. Composite foreign keys preserve owner identity across records. Original bytes live under `users/<owner>/uploads/<operation>/original.csv` in a private bucket. Titles and filenames never determine object identity.

Operations are pending, ready or failed. Database publication and S3 are not a distributed transaction: a pending operation is committed first; bytes are written and verified; dataset/file/version/context records publish together. An advisory lock fences retry and cleanup. A lost response is resolved by resubmitting the same key and payload. Original-byte and context records are immutable; no previous-layout loader exists.

`python -m causal_agent.server.ingestion cleanup` removes objects for unpublished operations older than 24 hours. It retains the idempotency record for safe retries and rechecks state under the same lock. Run this command periodically; no background scheduler is implied. Ready objects are never cleanup candidates. New-dataset deletion is outside this milestone.

## Worked walkthrough

Open `/new` → enter title, CSV and optional context → submit → validate → register pending operation → write and verify bytes → atomically publish relational records → show saved confirmation and download link. Success navigates to `/datasets/:datasetId`, which retrieves the saved record automatically, including after refresh. The owned library now lists the dataset. A failed save leaves the form intact. No conversation is created.

## Verification

[Ingestion tests](../../causal_agent/server/tests/test_ingestion.py) cover structural validation, limits, ownership, real PostgreSQL/S3 retrieval, retries, failed publication, cleanup, database immutability and the HTTP boundary. [Rendered form tests](../../web/src/pages/NewDataset.test.tsx) cover failure preservation, submission identity and progress. [Container smoke](../../docker/smoke.py) checks publication and retrieval across service replacement. See [deployment instructions](../production/containerization.md) for commands and observed limits. Authentication remains a development placeholder; this is not a public multi-user deployment.
