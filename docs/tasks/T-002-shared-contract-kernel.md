# T-002 — Shared contract kernel: canonical serialization, hashing, envelopes

Status: READY_FOR_OPUS
Owner: Opus (terminal `causal-final-de`)
Governing documents: `docs/product/SYSTEM-CONTRACT.md` (hash in `docs/LEDGER.md`), this spec.
Contract sections: §2 (identities), §3 (immutable artifact contract),
§8.2 (artifact commit — canonicalize/hash steps only), §15 criteria 2.
Evaluation mapping: `EV-SYS-001` (artifact validation, canonicalization,
hashing; the eval fixtures themselves are a later task — your unit tests must
cover the same risks at the unit layer).

Dependency: **T-001** provides `pyproject.toml`/`uv.lock`. You may draft code
immediately, but tests can only run once T-001's commit is on `main`. Wait for
it (check `git log`) before producing completion evidence. Do not create or
modify `pyproject.toml` yourself — if a dependency you need is missing, STOP
and return `FABLE REVISION REQUIRED`.

## Allowed-write files (exact; nothing else)

- `src/causal/shared/canonical.py`
- `src/causal/shared/contracts.py`
- `tests/shared/__init__.py` (empty, only if pytest requires it)
- `tests/shared/test_canonical.py`
- `tests/shared/test_contracts.py`

## Deliverable 1 — `canonical.py`

Canonical serialization fixing key ordering, number encoding, Unicode
normalization, timestamp format, and null handling (§3). Exact rules (Fable
decision D-004):

1. `canonical_bytes(payload: Mapping[str, object]) -> bytes`: UTF-8 JSON,
   keys sorted by Unicode code point at every nesting level, separators
   `(",", ":")` with no whitespace, `ensure_ascii=False`.
2. Every `str` value and key is NFC-normalized before encoding.
3. Integers: arbitrary precision, as-is. Floats: shortest round-trip repr
   (Python `json` default); `NaN`, `Infinity`, `-Infinity` raise
   `CanonicalizationError`. Booleans and `None` per JSON; `None` values are
   preserved, never dropped.
4. Timestamps are already-serialized strings at this layer; the models own
   datetime → string conversion (rule 5 below). `canonical_bytes` accepts only
   `Mapping`/`Sequence`/`str`/`int`/`float`/`bool`/`None` — any other type
   (including `datetime`, `bytes`, sets) raises `CanonicalizationError`.
5. `content_hash(payload) -> str`: lowercase hex SHA-256 (64 chars) of
   `canonical_bytes(payload)`.
6. `CanonicalizationError(ValueError)` with a stable `code` attribute:
   `non_finite_number` | `unsupported_type` | `duplicate_key_after_normalization`.
7. Two distinct keys in one mapping that become equal after NFC normalization
   raise `duplicate_key_after_normalization` (Fable decision D-006): silent
   collapse would make output depend on insertion order, violating rule 1.

## Deliverable 2 — `contracts.py`

Pydantic v2 (`pydantic==2.13.4`), every model `frozen=True`,
`extra="forbid"`, `strict=True` where practical.

1. `SensitivityClass` — `StrEnum`: `public`, `internal`, `restricted`,
   `secret_reference`.
2. `ArtifactRef` — `artifact_id: str`, `content_hash: str` (exactly
   `^[a-f0-9]{64}$`).
3. `ArtifactEnvelopeV1` — exactly the §3 table fields:
   `artifact_id`, `artifact_type`, `schema_version`, `content_hash`
   (64-hex pattern), `analysis_id`, `stage_run_id`, `producer_component`,
   `producer_version`, `parent_artifacts: tuple[ArtifactRef, ...]` (ordered),
   `sensitivity_class: SensitivityClass`, `created_at_utc: datetime`,
   `payload_locator: str`.
   - All identity/string fields: non-empty, ≤200 chars (identity format
     enforcement beyond that belongs to creation sites, Fable decision D-005).
   - `created_at_utc`: must be timezone-aware UTC; serializes to RFC 3339 with
     exactly 6 fractional digits and `Z` suffix (custom serializer).
   - `payload_locator`: exempt from the 200-char identity cap; its own rule is
     non-empty, ≤1024 chars, and must NOT contain `?`, `&`, or the substring
     `X-Amz-` (never a signed URL, §3). (Fable decision D-007.)
4. `HandoffManifestV1` — per §3 bullet list: `handoff_id`, `schema_version`,
   `analysis_id`, `producing_stage_run_id`, `receiving_stage_run_id`,
   `entries: tuple[ArtifactRef, ...]` (ordered, min length 1),
   `originating_outcome: str`, `approval_ids: tuple[str, ...]` (may be empty),
   `registry_version: str`, `compatibility_version: str`,
   `receiver_validation_result: str | None`,
   `receiver_error_codes: tuple[str, ...]`,
   `created_at_utc: datetime`, `accepted_at_utc: datetime | None`.
   Same timestamp and non-empty rules.
5. Each model exposes `canonical_payload() -> dict` (its serialized dict in
   canonical-ready form) so `content_hash(model.canonical_payload())` is
   well-defined and replay-stable.

No persistence, S3, PostgreSQL, event, registry, or trace code in this task.

## Line budget for this task

- `canonical.py` ≤ 150 logical lines; `contracts.py` ≤ 250 logical lines
  (both inside the shared package's 2,500 allocation; run
  `tools/budget_check.py` once T-001 lands).
- Tests combined ≤ 400 logical lines.

## Tests (required)

- Canonical: key-order permutations of the same nested dict give identical
  bytes and hash; NFC vs NFD strings normalize to the same bytes; NaN/Inf
  raise `non_finite_number`; `datetime`/`bytes`/`set` raise
  `unsupported_type`; `None` values survive; repeated hashing is identical
  (replay). One bounded hypothesis property: hash invariance under dict key
  insertion order.
- Contracts: valid envelope round-trip; extra field rejected; bad
  `content_hash` pattern rejected; naive datetime rejected; signed-URL-shaped
  `payload_locator` rejected; timestamp serializes to exactly-6-digit `Z`
  form; `canonical_payload()` of an envelope is hash-stable across two
  constructions with reordered inputs.

## Forbidden work

Any file outside the allowed-write list; touching `pyproject.toml`,
`uv.lock`, `docs/`, `CLAUDE.md`, `.claude/`; adding dependencies; persistence
or network code; `OperationalEventV1` or `ArtifactTypeRegistrationV1`
(later tasks); altering T-001's files.

## Completion evidence (return via SendMessage to the Fable session)

1. Hash-verification statement (SYSTEM-CONTRACT hash from `docs/LEDGER.md`,
   this spec's hash from the dispatch message).
2. `uv run pytest tests/shared` summary (all green).
3. `uv run ruff check src/causal/shared tests/shared` and
   `uv run mypy src/causal/shared` clean.
4. `tools/budget_check.py` JSON for your change set.
5. Changed-file list and git commit hash (commit only allowed files; message
   starts `T-002:`).
