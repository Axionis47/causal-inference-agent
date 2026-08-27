# HTTP API reference

## Authority and access

FastAPI generates the public contract from Python types.
The saved schema is [web/openapi.json](../web/openapi.json).
Frontend types are generated in [schema.d.ts](../web/src/lib/api/schema.d.ts).
Do not edit either generated file to change backend behavior.

The inventory below was generated from the application schema on 2026-11-10.
It lists operation names and payload types rather than duplicating every field.
Use OpenAPI for required properties, constraints, and response schemas.
The host development API also exposes `/docs` and `/openapi.json` on port 8008.
The container Caddy configuration proxies `/api`, not those documentation paths.

## Request conventions

All domain routes require a valid `causal_session` cookie.
Login, logout, and the database health check have their own access behavior.
The session cookie is HttpOnly and SameSite Strict.
The API resolves owner access for each domain record.
Another owner's record is not made available by knowing its UUID.

Mutations require `X-Causal-Client: web`.
If Origin is present, it must match a configured allowed origin.
Mutations also require a valid Content-Length.
Use the typed client or a conventional HTTP client that computes that header.

JSON requests use `application/json`.
Dataset and source uploads use `multipart/form-data`.
File downloads return bytes with a filename disposition.
Analysis events use `text/event-stream`.

| Rule | Contract |
| --- | --- |
| Submission replay | Reuse the same submission UUID for the same operation payload |
| Conflicting replay | A changed payload under the same identity conflicts |
| Context writes | Supply the expected parent revision |
| Chat sends | Supply the expected turn number |
| Analysis launch | Supply the expected context revision and readiness fingerprint |
| Approval | Supply the exact snapshot and current program hashes |
| Background response | HTTP 202 confirms submission, not scientific acceptance |

## Operations

Request type `none` means no request body.
Path and query parameters can still be required.
Multipart types are labeled explicitly.
Successful status codes do not replace the operation's returned lifecycle status.

<!-- api-inventory:start -->
| Method | Path | Request body | Success |
| --- | --- | --- | --- |
| `GET` | `/api/health` | `none` | 200 |
| `POST` | `/api/auth/login` | `LoginInput` | 200 |
| `GET` | `/api/auth/me` | `none` | 200 |
| `POST` | `/api/auth/logout` | `none` | 204 |
| `GET` | `/api/datasets` | `none` | 200 |
| `POST` | `/api/datasets` | `multipart form` | 201 |
| `GET` | `/api/datasets/{dataset_id}` | `none` | 200 |
| `GET` | `/api/datasets/{dataset_id}/preview` | `none` | 200 |
| `GET` | `/api/datasets/{dataset_id}/file` | `none` | 200 |
| `GET` | `/api/versions/{version_id}/context` | `none` | 200 |
| `POST` | `/api/versions/{version_id}/context/sources` | `multipart form` | 201 |
| `GET` | `/api/versions/{version_id}/context/sources/{source_id}` | `none` | 200 |
| `GET` | `/api/versions/{version_id}/context/sources/{source_id}/file` | `none` | 200 |
| `POST` | `/api/versions/{version_id}/context/preparations` | `PrepareInput` | 202 |
| `GET` | `/api/versions/{version_id}/context/preparations/{preparation_id}` | `none` | 200 |
| `POST` | `/api/versions/{version_id}/context/preparations/{preparation_id}/retry` | `none` | 202 |
| `GET` | `/api/versions/{version_id}/context/revisions/{revision_id}` | `none` | 200 |
| `POST` | `/api/versions/{version_id}/context/revisions` | `SaveInput` | 201 |
| `GET` | `/api/versions/{version_id}/context/revisions/{revision_id}/read` | `none` | 200 |
| `GET` | `/api/versions/{version_id}/conversations` | `none` | 200 |
| `POST` | `/api/versions/{version_id}/conversations` | `CreateConversation` | 201 |
| `GET` | `/api/conversations/{conversation_id}` | `none` | 200 |
| `POST` | `/api/conversations/{conversation_id}/turns` | `SendTurn` | 202 |
| `POST` | `/api/conversations/{conversation_id}/turns/{turn_id}/retry` | `none` | 202 |
| `POST` | `/api/conversations/{conversation_id}/turns/{turn_id}/apply-context` | `none` | 200 |
| `GET` | `/api/conversations/{conversation_id}/artifacts/{artifact_id}/files/{filename}` | `none` | 200 |
| `GET` | `/api/conversations/{conversation_id}/analysis-readiness` | `none` | 200 |
| `GET` | `/api/conversations/{conversation_id}/analyses` | `none` | 200 |
| `POST` | `/api/conversations/{conversation_id}/analyses` | `PrepareAnalysis` | 202 |
| `GET` | `/api/analyses/{run_id}` | `none` | 200 |
| `POST` | `/api/analyses/{run_id}/approve` | `ApproveAnalysis` | 202 |
| `POST` | `/api/analyses/{run_id}/retry` | `none` | 202 |
| `POST` | `/api/analyses/{run_id}/reject` | `RejectAnalysis` | 200 |
| `POST` | `/api/analyses/{run_id}/revise-program` | `RejectAnalysis` | 202 |
| `GET` | `/api/analyses/{run_id}/events` | `none` | 200 |
| `GET` | `/api/analyses/{run_id}/files/{filename}` | `none` | 200 |
| `GET` | `/api/analyses/{run_id}/checkpoints/{receipt_id}/input` | `none` | 200 |
| `GET` | `/api/analyses/{run_id}/agents/{agent_id}` | `none` | 200 |
| `GET` | `/api/analyses/{run_id}/provider-attempts/{attempt_id}` | `none` | 200 |

<!-- api-inventory:end -->

## Important payloads

| Type | Required purpose |
| --- | --- |
| `CreateConversation` | Submission identity, title, optional exact context revision |
| `SendTurn` | Submission identity, expected turn, text, mode, selected prior artifacts |
| `PrepareInput` | Selected context sources and expected base revision |
| `SaveInput` | Submission identity, expected revision, edits, optional brief |
| `PrepareAnalysis` | Submission identity, expected context revision, readiness fingerprint |
| `ApproveAnalysis` | Exact `snapshot_hash` and `program_hash` |
| `RejectAnalysis` | The same exact hashes and a review reason. Rejection stops the run; a program correction returns to computation with unchanged accepted study/design and spent allowances. Fresh approval is required. |

An optional launch question must equal the applied brief question.
A separate request field cannot replace the brief silently.
Readiness is read-only and computed in the backend.

## Query and read behavior

Dataset listing supports search and pagination.
Page numbers start at one.
Page size is at most 100.
The default list size is 20; the default preview size is 25.

Context keyed reads accept repeated `key` query parameters.
They bind the exact version and revision in the path.
An omitted key list reads the full context projection.
Dependency expansion is defined by the [Context contract](context-slice.md#exact-retrieval).

Analysis detail returns saved state, checkpoint views, tasks, programs, events, and available results.
A historical run can return `read_only: true` while retaining its original lifecycle status.
Consumers must honor both fields.

## SSE replay

Endpoint: `GET /api/analyses/{run_id}/events`.

The server starts after the larger of `after` and the `Last-Event-ID` header.
Both represent a run-local event sequence.
An invalid header cursor returns 400.

Each persisted event uses the SSE event name `checkpoint`.
Its JSON `kind` identifies the actual lifecycle event.
Do not treat the transport name as proof that a checkpoint passed.

```text
id: 17
event: checkpoint
data: {"sequence":17,"stage":"target","kind":"gate_accepted",...}
```

This frame is illustrative, not a recorded event.
The exact event object is `EventView` in OpenAPI.

The stream sends keepalive comments while it waits.
It sends a `settled` event at an approval pause or terminal state.
A connection can end after approximately 240 polling iterations.
Reconnect from the last received sequence.

Use the sequence to deduplicate client delivery.
Use the analysis view and accepted receipts for scientific progress.
The server releases the request database connection before the long-lived stream.
Each replay batch uses a short database session.

## Artifact access

Analysis downloads can expose these saved products when they exist:

- `frozen-context.json`, `context.json`, and `trajectory.json`.
- `snapshot.json` and `graph.json`.
- `checkpoint-<receipt-id>.json`.
- `program-<revision>.py` and `diagnostic-<index>.py`.
- `review-rejection-<revision>.json`.
- `execution-<revision>-<filename>` outputs.

Missing products return 404.
An unapproved run can have context and checkpoint artifacts without an execution result.
The chat artifact catalogue is a separate bounded selection from this download surface.
See [H13](component-contracts.md#h13-saved-analysis-to-chat).

## Errors

Errors normally return a JSON object with a `detail` field.
Validation errors use a concise application message.
Do not parse message prose as a stable error-code enum.

| HTTP status | Meaning |
| --- | --- |
| 400 | Invalid request detail, including an invalid event cursor |
| 401 | Missing, expired, or invalid authentication |
| 403 | Disallowed mutation header or Origin |
| 404 | Missing record or record outside the owner's scope |
| 409 | Stale state, incompatible contract, conflicting replay, or forbidden lifecycle action |
| 411 | Missing or invalid Content-Length for a mutation |
| 413 | Request exceeds its operation limit |
| 422 | Invalid submitted values, schema, or domain input |
| 503 | Database or file persistence is unavailable |

An HTTP error and a saved background-operation failure are different records.
Inspect the returned operation status after an accepted background request.
A provider 400 can be stored in a failed run even when the original launch returned 202.

## Contract changes

1. Change the owning Python contract and operation together.
2. Add or update relevant contract checks.
3. Regenerate the saved OpenAPI schema.
4. Regenerate frontend types.
5. Update consumers and this operation inventory.

```sh
uv run python -m backend.cli openapi
pnpm --dir web api:types
```

The [verification guide](verification.md) owns commands for broader validation.
