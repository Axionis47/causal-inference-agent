# T-005 — Persistence layer and artifact commit protocol

Status: ACCEPTED (single-session, D-008)
Contract sections: SYSTEM-CONTRACT §3 (immutable artifacts), §3.1 (registry
routing), §4 (run states), §8.1 (storage responsibilities), §8.2 (commit
protocol), §8.3 (integrity rules).

## Files

- `migrations/0001_shared_persistence.sql`
- `src/causal/shared/persistence.py`
- `tests/shared/conftest.py` (dockerized Postgres + MinIO fixtures)
- `tests/shared/test_persistence.py`

## Decisions

- D-017: dev/test object store is MinIO via Docker; product code depends only
  on an S3 client Protocol (boto3-compatible kwargs), so no boto3 import and
  no type stubs in `src/`. Production uses boto3 against any S3-compatible
  endpoint. Postgres runs as `postgres:18.6` (the §14 pin).
- D-018: legal run-state transition map (from the §4 diagram):
  `created→{tracing_preflight}`;
  `tracing_preflight→{running, failed_observability, failed}`;
  `running→{waiting_for_user, committing, completed, failed_observability, failed}`;
  `waiting_for_user→{running, failed}`;
  `committing→{running, completed, failed_observability, failed}`;
  `completed`, `failed`, `failed_observability` are terminal.
- D-019: `payload_locator` format is `objects/{content_hash}` — one
  content-addressed key space per bucket; the committer rejects an envelope
  whose locator disagrees with its hash.
- D-020: §8.2's LangSmith flush gate is DEFERRED to the tracing task; the
  committer emits the `artifact.committed` JSON event (caller-built) after
  reopen-validation. The seam: `ArtifactCommitter.commit(envelope, payload,
  event)` — when tracing lands, the flush-acknowledgement step slots in
  between reopen-validation and handoff visibility. Recorded so the deferral
  cannot silently become permanent.

## Deliverable — `migrations/0001_shared_persistence.sql`

Schema `causal`: `stage_runs` (id, analysis, stage CHECK, run_state CHECK,
timestamps), `artifacts` (all envelope scalar fields; content_hash and
sensitivity CHECKs; FK to stage_runs; indexes on analysis_id and
artifact_type), `artifact_parents` (ordered lineage: artifact FK,
parent_index, parent id+hash, PK (artifact_id, parent_index)).

## Deliverable — `persistence.py`

1. `PersistenceError(ValueError)` with stable `code`:
   `integrity_conflict` | `artifact_hash_mismatch` | `locator_mismatch` |
   `registry_violation` | `missing_required_parent` |
   `reopen_validation_failed` | `illegal_state_transition` |
   `unknown_stage_run`.
2. `S3ClientProtocol` (typing.Protocol): `head_object`, `put_object`,
   `get_object` with boto3 keyword shapes.
3. `ObjectStore(client, bucket)`: `locator_for(hash)`,
   `put_if_absent(hash, data) -> locator`, `get(locator) -> bytes`.
4. `apply_migrations(conn, migrations_dir)`: executes `*.sql` sorted.
5. `ProductStore(conn)`: `create_stage_run`, `transition_stage_run`
   (enforces D-018, terminal states refuse further transitions),
   `get_stage_run_state`, `load_envelope(artifact_id)` (row + ordered
   parents → `ArtifactEnvelopeV1`), plus the insert used by the committer.
6. `ArtifactCommitter(object_store, product_store, registry, emitter)`.
   `commit(envelope, payload, event)` implements §8.2:
   a. `content_hash(payload)` must equal `envelope.content_hash`
      (`artifact_hash_mismatch`); locator must match D-019
      (`locator_mismatch`).
   b. Registry checks (§3.1): type registered (`unsupported_schema` bubbles
      from T-004), producer/schema_version/sensitivity must match the
      registration (`registry_violation`); every `required_parent_type` must
      appear among the committed parents' types (`missing_required_parent`).
   c. Object put-if-absent, then the PostgreSQL transaction (artifact row +
      ordered parents).
   d. Replay rule (§8.2): same id + same hash → no-op returning the existing
      envelope; same id + different hash → `integrity_conflict`.
   e. Reopen: read the object back, re-verify SHA-256
      (`reopen_validation_failed`).
   f. Emit the caller-built `artifact.committed` event. (Trace flush: D-020.)

## Tests (dockerized; skipped when Docker is unavailable)

Object store roundtrip and idempotent re-put; migrations apply; stage-run
lifecycle including illegal-transition and terminal-state refusal; commit
happy path (envelope persisted, parents ordered, object readable, event line
emitted); replay no-op (no duplicate rows); id-reuse with different hash →
`integrity_conflict`; hash/locator mismatch codes; producer mismatch →
`registry_violation`; missing required parent → `missing_required_parent`;
unregistered type bubbles `unsupported_schema`; reopen failure via a
corrupting fake store.

## Budget

`persistence.py` ≤ 300 logical lines (shared); migration SQL ≤ 80
(declarative); conftest + tests ≤ 450 (tests scope).
