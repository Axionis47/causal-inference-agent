# T-006 — Handoff persistence and fail-closed acceptance gate

Status: ACCEPTED (single-session, D-008)
Contract sections: SYSTEM-CONTRACT §3 (handoff manifest, receiver rule),
§3.1 (allowed readers, destinations), §8.1 (handoff visibility in PostgreSQL),
§10.1 (`handoff.accepted` / `handoff.rejected`). Maps to EV-SYS-006.

## Files

- `migrations/0002_handoffs.sql`
- `src/causal/shared/handoff.py`
- `tests/shared/test_handoff.py`

## Decisions

- D-021: the receiver names its expected outcomes explicitly —
  `HandoffGate.accept(manifest, receiving_component, allowed_outcomes,
  event_factory)`. "Wrong outcome" is checked against that per-seam set, since
  which outcomes open a receiver is seam knowledge (e.g. PRD-004 opens only on
  `prepared`), not manifest knowledge.
- D-022: gate events are built by an `event_factory(result, error_codes)`
  callback so the emitter's D-016 eval-ID rule stays satisfiable by the
  caller; the gate emits exactly one `handoff.accepted` or `handoff.rejected`
  event per decision.

## Deliverable — `migrations/0002_handoffs.sql`

`causal.handoffs` (manifest scalar fields; `receiver_error_codes text[]`;
nullable `receiver_validation_result` and `accepted_at_utc`) and
`causal.handoff_entries` (handoff FK, entry_index, artifact id + hash,
PK (handoff_id, entry_index)).

## Deliverable — `handoff.py`

1. Stable failure codes: `missing_artifact` | `entry_hash_mismatch` |
   `missing_object` | `wrong_outcome` | `reader_not_allowed` |
   `incomplete_lineage` | `unsupported_version` | `unknown_handoff` |
   `duplicate_handoff`.
2. `HandoffStore(conn)`: `record(manifest)` (rejects duplicate handoff_id),
   `load(handoff_id) -> HandoffManifestV1`, `mark(handoff_id, result,
   error_codes, accepted_at)`.
3. `HandoffGate(object_store, product_store, handoff_store, registry,
   emitter)`. `accept(manifest, receiving_component, allowed_outcomes,
   event_factory)`:
   - `unsupported_version` when `manifest.schema_version != "handoff.v1"`;
   - `wrong_outcome` when `originating_outcome not in allowed_outcomes`;
   - per entry: artifact row exists (`missing_artifact`); its committed hash
     equals the entry hash (`entry_hash_mismatch`); the payload object loads
     and re-hashes to the entry hash (`missing_object`); `receiving_component`
     is in the type's registered readers (`reader_not_allowed`); every
     registered required parent type appears among the entry artifact's
     committed parents (`incomplete_lineage`);
   - collects ALL failure codes (fail-closed, complete diagnosis), records
     the manifest + `accepted`/`rejected` marks in PostgreSQL, emits the
     factory-built event, and returns an `accepted: bool` plus sorted codes.

## Tests (dockerized; same fixtures as T-005)

Happy path (entries committed via T-005's committer → gate accepts, rows
marked, `handoff.accepted` emitted); each failure code has one test; multiple
failures all reported; duplicate handoff_id rejected; `load` of unknown
handoff raises.

## Budget

`handoff.py` ≤ 220 logical lines (shared); migration ≤ 40 (declarative);
tests ≤ 350.
