"""Nested LangSmith application traces, credential redaction, and delivery gates."""

from __future__ import annotations

import logging
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, cast

if TYPE_CHECKING:  # the client is imported lazily so construction stays cheap
    from langsmith import Client

__all__ = [
    "API_KEY_ENV_VARS", "FLUSH_UNACKNOWLEDGED", "PREFLIGHT_FAILED",
    "REDACTION_PATTERNS_V1", "REDACTION_POLICY_VERSION", "SAFE_METADATA_KEYS_V1",
    "LangSmithTracer", "ObservabilityError", "TraceRedactorV1", "TraceSpanProtocol",
    "TracerProtocol", "sanitize_diagnostic_event", "trace_span",
]

PREFLIGHT_FAILED: Final = "preflight_failed"
FLUSH_UNACKNOWLEDGED: Final = "flush_unacknowledged"

REDACTION_POLICY_VERSION: Final = "trace-redaction.v1"

# langsmith 0.11.0 reads the key from either namespace (`langsmith.utils.get_env_var`).
API_KEY_ENV_VARS: Final[tuple[str, ...]] = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")

DEFAULT_FLUSH_TIMEOUT_SECONDS: Final = 30.0
_LOGGER = logging.getLogger(__name__)
_CREDENTIAL_KEYS = frozenset({
    "apikey", "token", "accesstoken", "refreshtoken", "secret", "clientsecret",
    "password", "authorization", "proxyauthorization", "privatekey", "secretaccesskey",
})
_QUOTED_CREDENTIAL = re.compile(
    r'''(?i)(["'](?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|'''
    r'''client[_-]?secret|password|authorization|private[_-]?key)["']\s*:\s*)'''
    r'''("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')''')

# Applied in order; each replacement text is inert for every later pattern.
REDACTION_PATTERNS_V1: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("signed_url", re.compile(r"X-Amz-[A-Za-z-]+=\S+")),
    ("bearer", re.compile(r"Bearer\s+\S+")),
    ("authority_credentials", re.compile(r"[a-z][a-z0-9+.-]*://[^/\s:]+:[^@\s]+@")),
    ("api_key_assignment", re.compile(r"(?i)(api[_-]?key|token|secret)\s*[=:]\s*\S+")),
)

# Scalar metadata remains allowlisted; full application content lives in inputs/outputs.
SAFE_METADATA_KEYS_V1: Final[frozenset[str]] = frozenset(
    {
        "environment", "analysis_id", "stage_run_id", "graph_thread_id", "task_id", "attempt_id",
        "parent_event_id", "graph_version", "schema_version", "agent_type", "prompt_version", "model_profile_version",
        "tool_registry_version", "validator_registry_version", "selected_method", "method_pack_version",
        "selected_csv_artifact_id", "validation_status", "error_code", "causal_graph_view_artifact_id",
        "renderer_version", "causal_graph_view_validation_status", "correction_attempt_count",
        "envelope_id", "envelope_hash", "prompt_hash", "prompt_characters",
        "response_schema_hash", "model_id", "seed", "physical_attempts",
        "input_tokens", "output_tokens", "thinking_tokens", "total_tokens", "finish_reason",
        "evaluation_run_id", "evaluation_case_id", "evaluation_mode", "final_outcome",
    }
)

# Exact report-safe proof surface for the bounded model→diagnostic→revision loop. Each event is
# normalized to one decision source; arbitrary safe_dimensions never cross this wall.
_DIAGNOSTIC_EVENTS: Final[dict[str, tuple[str, tuple[str, ...]]]] = {
    "agent.diagnostic_requested": ("llm_decision", ("task_id", "attempt_number", "diagnostic_id", "remaining_tool_calls", "request_hash")),
    "diagnostic.completed": ("deterministic_normalization", ("diagnostic_id", "status", "result_hash", "warning_count", "used_row_count")),
    "agent.design_revised": ("llm_decision", ("task_id", "attempt_number", "prior_proposal_hash", "revised_proposal_hash", "triggering_diagnostic_ids")),
    "agent.escalated": ("compiler_failure", ("task_id", "error_code", "responsible_actor", "exhausted_budget")),
}
_DIAGNOSTIC_HASH_FIELDS: Final = frozenset({"request_hash", "result_hash", "prior_proposal_hash", "revised_proposal_hash"})
_DIAGNOSTIC_COUNT_FIELDS: Final = frozenset({"attempt_number", "remaining_tool_calls", "warning_count", "used_row_count"})
_SAFE_ID: Final = re.compile(r"^[A-Za-z0-9_.:-]+(?:,[A-Za-z0-9_.:-]+)*$")
_SAFE_HASH: Final = re.compile(r"^[a-f0-9]{64}$")


def sanitize_diagnostic_event(event: Mapping[str, object]) -> dict[str, object] | None:
    """Flatten one loop event onto its closed scalar/hash surface; never copy observations."""
    name = event.get("event_name")
    if not isinstance(name, str) or name not in _DIAGNOSTIC_EVENTS:
        return None
    source, fields = _DIAGNOSTIC_EVENTS[name]
    dimensions = event.get("safe_dimensions")
    raw = dict(dimensions) if isinstance(dimensions, Mapping) else {}
    raw.update(event)
    safe: dict[str, object] = {field: value for field in fields
        if (value := raw.get(field)) is not None and (
        (field == "exhausted_budget" and isinstance(value, bool)) or
        (field in _DIAGNOSTIC_COUNT_FIELDS and isinstance(value, int)
         and not isinstance(value, bool) and value >= 0) or
        (isinstance(value, str) and len(value) <= 200 and _SAFE_ID.fullmatch(value)
         and (field not in _DIAGNOSTIC_HASH_FIELDS or _SAFE_HASH.fullmatch(value))))}
    return {"event_name": name, "decision_source": source} | safe | {
        "complete": all(field in safe for field in fields)}


class ObservabilityError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class TracerProtocol(Protocol):
    """Fail-closed preflight, safe gateway spans, and delivery flush (SC §10.2)."""

    def preflight(self) -> None: ...

    def flush(self) -> None: ...

    def start_gateway_span(self, metadata: Mapping[str, object]) -> TraceSpanProtocol: ...


class TraceSpanProtocol(Protocol):
    def finish(self, outputs: Mapping[str, object] | None = None,
               error_code: str | None = None) -> None: ...


class _LangSmithSpan:
    def __init__(self, client: Any, run_id: uuid.UUID, redactor: TraceRedactorV1,
                 *, name: str, trace_id: uuid.UUID, dotted_order: str,
                 metadata: Mapping[str, object], gateway: bool = False) -> None:
        self._client, self._run_id, self._redactor = client, run_id, redactor
        self.trace_id, self.dotted_order = trace_id, dotted_order
        self._name, self._metadata, self._gateway = name, dict(metadata), gateway
        self._finished = False

    def finish(self, outputs: Mapping[str, object] | None = None,
               error_code: str | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        raw = dict(outputs or {})
        safe_metadata = self._metadata | self._redactor.redact_metadata(
            raw | ({"error_code": error_code} if error_code else {}))
        if self._gateway:
            raw = self._redactor.redact_metadata(raw) | {
                key: raw[key] for key in ("model_output", "provider_summary", "decision_summary")
                if key in raw}
            if "model_output" in raw:
                raw["messages"] = [{"role": "assistant", "content": raw["model_output"]}]
        safe_error = self._redactor.redact_text(error_code) if error_code else None
        try:
            safe = self._redactor.redact_content(raw)
            self._client.update_run(
                self._run_id, end_time=datetime.now(UTC), outputs=safe,
                error=safe_error, extra={"metadata": safe_metadata},
                trace_id=self.trace_id, dotted_order=self.dotted_order)
        except Exception as error:
            raise ObservabilityError(
                self._redactor.redact_text(f"LangSmith span finish failed: {error!r}"),
                FLUSH_UNACKNOWLEDGED
            ) from error
        _LOGGER.info("trace.span.finished", extra={"trace_span": {
            "name": self._name, "run_id": str(self._run_id), "error_code": safe_error}})


class TraceRedactorV1:
    redaction_policy_version: Literal["trace-redaction.v1"] = REDACTION_POLICY_VERSION

    def redact_text(self, text: str) -> str:
        text = _QUOTED_CREDENTIAL.sub(r'\1"[REDACTED:credential]"', text)
        for name, pattern in REDACTION_PATTERNS_V1:
            text = pattern.sub(f"[REDACTED:{name}]", text)
        return text

    def redact_metadata(
        self, mapping: Mapping[str, object]
    ) -> dict[str, str | int | float | bool]:
        return {key: self.redact_text(value) if isinstance(value, str) else value
                for key, value in mapping.items() if key in SAFE_METADATA_KEYS_V1
                and isinstance(value, str | bool | int | float)}

    def redact_content(self, mapping: Mapping[str, object]) -> dict[str, object]:
        """Keep complete JSON application content; recursively mask credential values.

        Callers serialize models before tracing. No study-data fields or prose are omitted,
        and no length limit silently replaces application content with a hash or preview.
        """
        def clean(value: object) -> object:
            if isinstance(value, Mapping):
                result: dict[str, object] = {}
                image = str(value.get("mime_type", "")).startswith("image/")
                for key, item in value.items():
                    normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
                    if normalized in _CREDENTIAL_KEYS:
                        safe: object = "[REDACTED:credential]"
                    elif key == "base64" and image and isinstance(item, str):
                        safe = item  # Opaque media bytes must not be altered as if they were prose.
                    else:
                        safe = clean(item)
                    result[self.redact_text(str(key))] = safe
                return result
            if isinstance(value, list | tuple):
                return [clean(item) for item in value]
            if isinstance(value, str):
                return self.redact_text(value)
            if value is None or isinstance(value, bool | int | float):
                return value
            raise TypeError(f"trace content must be JSON-compatible: {type(value).__name__}")

        return cast(dict[str, object], clean(mapping))


class _NoopSpan:
    def finish(self, outputs: Mapping[str, object] | None = None,
               error_code: str | None = None) -> None:
        pass


@contextmanager
def trace_span(tracer: TracerProtocol | None, name: str, *,
               run_type: Literal["chain", "tool", "llm"] = "chain",
               inputs: Mapping[str, object] | None = None,
               metadata: Mapping[str, object] | None = None) -> Iterator[TraceSpanProtocol]:
    """Use rich tracing when supported; preserve optional/legacy tracer compatibility."""
    factory = getattr(tracer, "span", None)
    if callable(factory):
        with factory(name, run_type=run_type, inputs=inputs, metadata=metadata) as span:
            yield span
    else:
        yield _NoopSpan()


class LangSmithTracer:
    def __init__(
        self,
        project: str,
        environment: str,
        redactor: TraceRedactorV1,
        *,
        client_factory: Callable[[Callable[[Exception], None]], Any] | None = None,
        flush_timeout_seconds: float = DEFAULT_FLUSH_TIMEOUT_SECONDS,
    ) -> None:
        self._project = project
        self._environment = environment
        self._redactor = redactor
        self._client_factory = client_factory
        self._flush_timeout_seconds = flush_timeout_seconds
        self._delivery_errors: list[Exception] = []
        self._client_obj: Any = None
        self._active: ContextVar[_LangSmithSpan | None] = ContextVar(
            f"causal_trace_{id(self)}", default=None)

    def _record_delivery_error(self, error: Exception) -> None:
        self._delivery_errors.append(error)

    def _build_client(self) -> Client:
        # Imported here, not at module scope: constructing the tracer must stay cheap.
        from langsmith import Client

        # Content has already crossed our credential redactor. Do not let ambient SDK
        # defaults silently hide or sample away the application trace the caller requested.
        return Client(tracing_error_callback=self._record_delivery_error,
                      hide_inputs=False, hide_outputs=False, tracing_sampling_rate=1.0)

    def _client(self) -> Any:
        if self._client_obj is None:
            self._client_obj = (
                self._build_client()
                if self._client_factory is None
                else self._client_factory(self._record_delivery_error)
            )
        return self._client_obj

    def preflight(self) -> None:
        if not any(os.environ.get(name, "").strip() for name in API_KEY_ENV_VARS):
            raise ObservabilityError(
                f"no LangSmith API key in {list(API_KEY_ENV_VARS)}", PREFLIGHT_FAILED
            )
        try:
            self._client().has_project(self._project)
        except Exception as error:
            raise ObservabilityError(
                self._redactor.redact_text(f"LangSmith preflight call failed: {error!r}"),
                PREFLIGHT_FAILED
            ) from error

    def start_gateway_span(self, metadata: Mapping[str, object]) -> TraceSpanProtocol:
        safe = self._redactor.redact_metadata(
            dict(metadata) | {"environment": self._environment})
        inputs: dict[str, object] = dict(safe)
        inputs.update({key: metadata[key] for key in ("prompt", "response_schema", "images")
                       if key in metadata})
        if "prompt" in inputs:
            content: object = inputs["prompt"]
            images = inputs.get("images")
            if isinstance(images, list) and images:
                content = [{"type": "text", "text": inputs["prompt"]}, *(
                    {"type": "image", "base64": image["base64"], "mime_type": image["mime_type"]}
                    for image in images)]
            inputs["messages"] = [{"role": "user", "content": content}]
        return self._start_span("model.gateway", "llm", inputs, safe, gateway=True)

    @contextmanager
    def span(self, name: str, *, run_type: Literal["chain", "tool", "llm"] = "chain",
             inputs: Mapping[str, object] | None = None,
             metadata: Mapping[str, object] | None = None) -> Iterator[TraceSpanProtocol]:
        """Nest a node/tool/model span; finish with full outputs and explicit decision_summary.

        Summaries describe application decisions or exposed provider summaries. This API
        does not request, reconstruct, or claim access to private model reasoning.
        """
        span = self._start_span(name, run_type, inputs or {}, metadata or {})
        token = self._active.set(span)
        try:
            yield span
        except BaseException as error:
            span.finish(error_code=str(getattr(error, "code", type(error).__name__)))
            raise
        else:
            span.finish()
        finally:
            self._active.reset(token)

    def _start_span(self, name: str, run_type: Literal["chain", "tool", "llm"],
                    inputs: Mapping[str, object], metadata: Mapping[str, object], *,
                    gateway: bool = False) -> _LangSmithSpan:
        safe = self._redactor.redact_metadata(
            dict(metadata) | {"environment": self._environment})
        safe_name = self._redactor.redact_text(name)
        run_id = uuid.uuid4()
        start_time = datetime.now(UTC)
        parent = self._active.get()
        trace_id = parent.trace_id if parent else run_id
        order = f"{start_time.strftime('%Y%m%dT%H%M%S%fZ')}{run_id}"
        dotted_order = f"{parent.dotted_order}.{order}" if parent else order
        try:
            self._client().create_run(
                safe_name, self._redactor.redact_content(inputs), run_type,
                id=run_id, project_name=self._project, start_time=start_time,
                extra={"metadata": safe}, trace_id=trace_id, dotted_order=dotted_order,
                parent_run_id=parent._run_id if parent else None)
        except Exception as error:
            raise ObservabilityError(
                self._redactor.redact_text(f"LangSmith span start failed: {error!r}"),
                PREFLIGHT_FAILED) from error
        _LOGGER.info("trace.span.started", extra={"trace_span": {
            "name": safe_name, "run_id": str(run_id), "run_type": run_type}})
        return _LangSmithSpan(self._client(), run_id, self._redactor, name=safe_name,
                             trace_id=trace_id, dotted_order=dotted_order,
                             metadata=safe, gateway=gateway)

    def flush(self) -> None:
        client = self._client()
        try:
            client.flush(timeout=self._flush_timeout_seconds)
        except Exception as error:
            raise ObservabilityError(
                self._redactor.redact_text(f"LangSmith flush failed: {error!r}"),
                FLUSH_UNACKNOWLEDGED
            ) from error
        # langsmith 0.11.0 has no per-batch acknowledgement: `Client.flush()` drains the
        # tracing queue and never raises on a rejected batch. Delivery failure is therefore
        # detected through the `tracing_error_callback` the client is built with (invoked
        # once per exhausted ingest attempt), and a queue still holding unfinished tasks
        # after the flush timeout counts as unacknowledged. Limit: batches the client never
        # enqueued (e.g. dropped by sampling) are invisible to both checks.
        queue = client.tracing_queue
        if queue is not None and queue.unfinished_tasks:
            raise ObservabilityError(
                f"{queue.unfinished_tasks} trace batches undelivered after flush",
                FLUSH_UNACKNOWLEDGED,
            )
        if self._delivery_errors:
            first = self._delivery_errors[0]
            self._delivery_errors.clear()
            raise ObservabilityError(
                self._redactor.redact_text(f"LangSmith rejected a trace batch: {first!r}"),
                FLUSH_UNACKNOWLEDGED
            )
