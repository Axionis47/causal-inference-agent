# T-010 — Shared Vertex gateway and LangSmith tracing

Status: frozen for implementation
Owning scope: shared (SC §10.2–§10.4; PRD-002 §20); closes the D-020 revisit
Depends on: T-009

## 1. Deliverables

1. `src/causal/shared/gateway.py` — the one Vertex model gateway (SC §10.4).
2. `src/causal/shared/tracing.py` — LangSmith preflight/span/flush plus the trace redactor
   (SC §10.2, §10.3).
3. `src/causal/shared/persistence.py` — commit-protocol flush gate: `ArtifactCommitter` accepts
   an optional tracer; when present, a commit is complete only after an acknowledged flush
   (SC §8.2). `None` preserves current behavior until runtime wiring (T-014). This closes D-020's
   deferral: the gate exists; wiring it everywhere is runtime composition.
4. Tests: `tests/shared/test_gateway.py`, `tests/shared/test_tracing.py`, plus env-gated live
   smoke tests (`RUN_LIVE_VERTEX=1`, `RUN_LIVE_LANGSMITH=1`; skipped by default).

## 2. gateway.py

- `VertexModelProfileV1` (frozen pydantic): `profile_version` Literal `vertex-model-profile.v1`,
  `sdk` = `google-genai==2.19.0`, `api_surface` = `v1`, `model_id` = `gemini-2.5-flash`,
  `location` = `us-central1`, `authentication` = `adc`, `temperature` = 0.0,
  `candidate_count` = 1, `thinking_budget_tokens` = 8192, `max_output_tokens` = 16384,
  `response_mime_type` = `application/json`, `automatic_function_calling` = False,
  `sdk_retries` = False. Module constant `VERTEX_PROFILE_V1` is the only instance in V1.
- `derive_seed(task_id: str) -> int`: first 8 hex chars of sha256(task_id utf-8) as unsigned
  32-bit int; deterministic and reused for the same task (SC §10.4).
- `GenerationSettingsV1` (frozen): the exact per-call knobs the transport must apply
  (temperature, candidate_count, seed, thinking budget, max output, mime type, response_schema:
  dict, automatic_function_calling: bool).
- `TransportResponse` (frozen dataclass or model): `text: str`,
  `token_usage: dict[str, int]` (keys from D-015: input/output/thinking/total),
  `finish_reason: str`.
- `TransportError(Exception)`: `code: str`, `retryable: bool`. The transport translates provider
  exceptions; the gateway never inspects provider types.
- `ModelTransportProtocol` (Protocol): `generate(model_id: str, prompt: str,
  settings: GenerationSettingsV1) -> TransportResponse`.
- `GatewayError(Exception)`: stable codes `provider_safety_rejection`, `model_unavailable`,
  `invalid_authentication`, `permission_denied`, `quota_exhausted`,
  `unsupported_structured_output`, `transient_exhausted`. All raise blockers at the caller; the
  gateway never substitutes model, provider, prompt, or config (SC §7.1).
- `GatewayResultV1`: `text`, `parsed: dict[str, object] | None` (json.loads of text; `None` when
  the text is not valid JSON — schema failure handling belongs to the harness correction loop,
  not the gateway), `token_usage`, `attempts: int`, `seed: int`.
- `VertexGateway(transport, profile, emitter, clock)`:
  `invoke(envelope: AgentTaskEnvelopeV1, prompt: str, response_schema: dict) -> GatewayResultV1`.
  - Builds `GenerationSettingsV1` strictly from the frozen profile + `derive_seed(task_id)`.
  - Transient rule (SC §7): up to `envelope.budgets.transient_attempt_budget` total physical
    attempts, only on `TransportError(retryable=True)`; emits `retry.scheduled` between attempts
    and `retry.exhausted` before raising `GatewayError("transient_exhausted")`. Non-retryable
    transport errors map 1:1 to `GatewayError(code)` with no further attempt. Events are built
    with `causal.shared.events.build_event` using the envelope identities and
    `required_eval_ids=("EV-SYS-003",)`.
- `GenAiTransport` — the live adapter over the installed `google-genai==2.19.0`:
  lazily constructs the client in Vertex mode (ADC; project resolved by the SDK; never logged),
  applies every `GenerationSettingsV1` knob (structured output via response schema + JSON mime
  type, thinking budget, seed, candidate count, AFC disabled, SDK retries disabled), and
  translates provider exceptions to `TransportError` (auth/permission/safety/quota → retryable
  False; 5xx/timeout/connection → retryable True). IMPLEMENTATION RULE: introspect the installed
  package (`uv run python -c "from google import genai; ..."`, `inspect.signature`) and use only
  APIs that exist in the installed 2.19.0 — no guessed keyword arguments.

## 3. tracing.py

- `ObservabilityError(Exception)`: `code` in {`preflight_failed`, `flush_unacknowledged`}.
- `TracerProtocol` (Protocol): `preflight() -> None`, `span(name, *, run_type: str,
  metadata: dict[str, object]) -> AbstractContextManager[str]` (yields span id),
  `flush() -> None`. Every failure surfaces as `ObservabilityError`; there is no fallback,
  queue, or spool (SC §10.2).
- `TraceRedactorV1`: `redaction_policy_version` Literal `trace-redaction.v1`.
  - `redact_metadata(mapping) -> dict[str, str | int | float | bool]`: allowlist-only — keys
    outside the safe-metadata allowlist (PRD-002 §20.3 list, module constant) are dropped;
    non-scalar values are dropped.
  - `redact_text(text) -> str`: pattern scrub replacing matches with `[REDACTED:<class>]`.
    Minimum pattern classes (module-level, versioned with the policy): `aws_key`
    (`AKIA[0-9A-Z]{16}`), `signed_url` (`X-Amz-[A-Za-z-]+=\S+`), `bearer` (`Bearer\s+\S+`),
    `authority_credentials` (`[a-z][a-z0-9+.-]*://[^/\s:]+:[^@\s]+@` — any URL userinfo
    password), `api_key_assignment` (`(?i)(api[_-]?key|token|secret)\s*[=:]\s*\S+`).
- `LangSmithTracer(project, environment, redactor)` — wraps the installed `langsmith==0.11.0`
  client. `preflight()` requires the API key to be present in the environment and one cheap
  authenticated call to succeed; absence or failure → `ObservabilityError("preflight_failed")`
  (fail closed; D-043). `span()` creates/closes a run with redacted metadata plus the
  environment/project tags. `flush()` forces client delivery and raises on any unacknowledged
  batch. Same introspection rule as the transport: use only APIs present in the installed
  package.
- Persistence delta: `ArtifactCommitter.__init__` gains `tracer: TracerProtocol | None = None`;
  after the existing event emission, a set tracer must `flush()`; `ObservabilityError`
  propagates (callers map it to `failed_observability` and stop; committed artifacts are
  preserved — SC §8.2). No other behavior change; existing tests must pass unmodified.

## 4. Tests

- Gateway (fake transport recording settings): profile knobs land exactly (temperature 0.0,
  candidate 1, thinking 8192, max 16384, JSON mime, AFC off, schema passthrough); seed stable
  per task_id, distinct across task_ids, < 2**32; retryable failure ×2 then success → attempts 3
  and two `retry.scheduled` events; exhaustion → `retry.exhausted` + `transient_exhausted`;
  non-retryable auth error → one attempt, `invalid_authentication`; non-JSON text → parsed None;
  token usage passthrough.
- Tracing: redactor canaries (one fixture per pattern class must not survive `redact_text`;
  clean text passes unchanged); metadata allowlist keeps safe keys and drops unknown/non-scalar;
  fake failing tracer: `preflight` and `flush` raise typed codes.
- Committer flush gate (dockerized, existing fixtures): commit with a fake acknowledged tracer
  succeeds; commit with a fake tracer whose `flush` fails raises `ObservabilityError` while the
  artifact row and object remain committed; committer without tracer behaves as before.
- Live smokes: one `RUN_LIVE_VERTEX=1` structured-output call asserting valid JSON against a toy
  schema; one `RUN_LIVE_LANGSMITH=1` preflight+span+flush round trip. Both `pytest.mark.skipif`
  on the env var by default.

## 5. Budgets and acceptance

Targets: gateway ≤ 260, tracing ≤ 240, persistence delta ≤ +15 (shared scope stays ≤ 2,500);
tests ≤ 500 new logical lines. Acceptance identical to T-009 §7 (suite green, ruff, mypy
--strict, budget gate, ledger + checkpoint commit).
