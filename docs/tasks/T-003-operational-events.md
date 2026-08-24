# T-003 — OperationalEventV1 and NDJSON event emitter

Status: ACCEPTED (single-session, D-008)
Contract sections: SYSTEM-CONTRACT §10.1 (event schema and required names),
§8.1 (JSON application logs), §15 criterion 8.

## Files

- `src/causal/shared/events.py`
- `tests/shared/test_events.py`

## Deliverable — `events.py`

1. `Severity` StrEnum: `debug`, `info`, `warning`, `error`.
2. `Stage` StrEnum: `intake`, `design`, `preparation`, `estimation`,
   `presentation`, `system` (D-013: `system` covers shared/CLI/runtime
   components that act outside one stage).
3. `EVENT_NAMES_V1`: frozenset of exactly the 28 §10.1 required names.
4. `OperationalEventV1` (frozen, extra=forbid, strict; reuses `Identity`,
   `Sha256Hex`, `UtcTimestamp`, `ArtifactRef` from the T-002 kernel):
   - `schema_version: Literal["operational-event.v1"]`
   - `occurred_at_utc: UtcTimestamp`; `severity: Severity`
   - `event_name`: dotted lowercase pattern `^[a-z0-9_]+(\.[a-z0-9_]+)+$`
     (registry membership is enforced by the emitter, not the model)
   - `event_id: Identity`; `parent_event_id: Identity | None`
   - `analysis_id: Identity`; `stage: Stage`; `stage_run_id: Identity`
   - `graph_thread_id`, `task_id`, `attempt_id`: `Identity | None`;
     `attempt_number: int | None` (≥1)
   - `component_id`, `component_version`: `Identity`
   - `versions: dict[str, Identity]` with keys restricted to
     {`model`, `prompt`, `tool`, `registry`, `schema`, `validator`,
     `compiler`, `renderer`} (D-014)
   - `status: Identity | None`; `error_code: Identity | None`;
     `retryable: bool | None`
   - `duration_ms: float | None` (≥0); `token_usage: dict[str, int]` with keys
     restricted to {`input`, `output`, `thinking`, `total`}, values ≥0
     (D-015); `cost: float | None` (≥0)
   - `artifact_refs: tuple[ArtifactRef, ...]`
   - `required_eval_ids: tuple[Identity, ...]`
   - evaluation fields, all optional: `evaluation_run_id`,
     `evaluation_case_id`, `evaluation_fixture_hash: Sha256Hex | None`,
     `evaluator_version`, `evaluation_gate_status`
   - `exception_class: Identity | None`;
     `exception_fingerprint: Identity | None`
   - `safe_dimensions: dict[str, str | int | float | bool]` (scalars only —
     enforced by type)
5. `EventEmitterError(ValueError)` with stable `code`:
   `unregistered_event_name` | `missing_required_eval_ids`.
6. `EventEmitter(sink: TextIO, registered_names: frozenset[str] =
   EVENT_NAMES_V1)`. `emit(event)`:
   - raises `unregistered_event_name` if `event.event_name` not registered;
   - raises `missing_required_eval_ids` when the name starts with `task.`,
     `agent.`, `tool.`, or `handoff.` and `required_eval_ids` is empty
     (D-016, §10.1 `required_eval_ids` row);
   - writes exactly one canonical-JSON line (via `canonical_bytes` of
     `model_dump(mode="json")`) plus `\n`, then flushes.

## Tests

Valid event round-trip; extra field rejected; bad event-name shape rejected by
the model; unregistered name rejected by the emitter with its code; a `task.*`
event without eval IDs rejected with its code; a `stage.*` event without eval
IDs accepted; emitted line parses back to the same payload and ends with one
newline; emission is deterministic (same event → identical line); versions/
token_usage key restriction enforced.

## Budget

`events.py` ≤ 200 logical lines (shared allocation); tests ≤ 250.
