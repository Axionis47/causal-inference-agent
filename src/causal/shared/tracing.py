"""LangSmith preflight/span/flush and the V1 trace redactor (SC §10.2, §10.3; PRD-002 §20)."""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

if TYPE_CHECKING:  # the client is imported lazily so construction stays cheap
    from langsmith import Client

__all__ = [
    "API_KEY_ENV_VARS",
    "FLUSH_UNACKNOWLEDGED",
    "PREFLIGHT_FAILED",
    "REDACTION_PATTERNS_V1",
    "REDACTION_POLICY_VERSION",
    "SAFE_METADATA_KEYS_V1",
    "LangSmithTracer",
    "ObservabilityError",
    "TraceRedactorV1",
    "TracerProtocol",
]

PREFLIGHT_FAILED: Final = "preflight_failed"
FLUSH_UNACKNOWLEDGED: Final = "flush_unacknowledged"

REDACTION_POLICY_VERSION: Final = "trace-redaction.v1"

# langsmith 0.11.0 reads the key from either namespace (`langsmith.utils.get_env_var`).
API_KEY_ENV_VARS: Final[tuple[str, ...]] = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")

DEFAULT_FLUSH_TIMEOUT_SECONDS: Final = 30.0

# Applied in order; each replacement text is inert for every later pattern.
REDACTION_PATTERNS_V1: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("signed_url", re.compile(r"X-Amz-[A-Za-z-]+=\S+")),
    ("bearer", re.compile(r"Bearer\s+\S+")),
    ("authority_credentials", re.compile(r"[a-z][a-z0-9+.-]*://[^/\s:]+:[^@\s]+@")),
    ("api_key_assignment", re.compile(r"(?i)(api[_-]?key|token|secret)\s*[=:]\s*\S+")),
)

# The PRD-002 §20.3 safe-metadata list; nothing outside it reaches LangSmith.
SAFE_METADATA_KEYS_V1: Final[frozenset[str]] = frozenset(
    {
        "environment",
        "analysis_id", "stage_run_id", "graph_thread_id", "task_id", "attempt_id",
        "parent_event_id",
        "graph_version", "schema_version",
        "agent_type",
        "prompt_version", "model_profile_version",
        "tool_registry_version", "validator_registry_version",
        "selected_method", "method_pack_version",
        "selected_csv_artifact_id",
        "validation_status", "error_code",
        "causal_graph_view_artifact_id", "renderer_version",
        "causal_graph_view_validation_status",
        "correction_attempt_count",
        "final_outcome",
    }
)


class ObservabilityError(Exception):
    """Tracing failed. `code` is `preflight_failed` or `flush_unacknowledged`."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class TracerProtocol(Protocol):
    """Preflight, span, flush; every failure is an ObservabilityError (SC §10.2)."""

    def preflight(self) -> None: ...

    def span(
        self, name: str, *, run_type: str, metadata: dict[str, object]
    ) -> AbstractContextManager[str]: ...

    def flush(self) -> None: ...


class TraceRedactorV1:
    """Allowlist metadata and scrub text before anything leaves the process (PRD-002 §20)."""

    redaction_policy_version: Literal["trace-redaction.v1"] = REDACTION_POLICY_VERSION

    def redact_text(self, text: str) -> str:
        for name, pattern in REDACTION_PATTERNS_V1:
            text = pattern.sub(f"[REDACTED:{name}]", text)
        return text

    def redact_metadata(
        self, mapping: Mapping[str, object]
    ) -> dict[str, str | int | float | bool]:
        safe: dict[str, str | int | float | bool] = {}
        for key, value in mapping.items():
            if key not in SAFE_METADATA_KEYS_V1:
                continue
            if isinstance(value, str):
                safe[key] = self.redact_text(value)
            elif isinstance(value, bool | int | float):
                safe[key] = value
        return safe


class LangSmithTracer:
    """The one LangSmith tracer: fail-closed preflight, redacted spans, acknowledged flush."""

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

    def _record_delivery_error(self, error: Exception) -> None:
        self._delivery_errors.append(error)

    def _build_client(self) -> Client:
        # Imported here, not at module scope: constructing the tracer must stay cheap.
        from langsmith import Client

        return Client(tracing_error_callback=self._record_delivery_error)

    def _client(self) -> Any:
        if self._client_obj is None:
            self._client_obj = (
                self._build_client()
                if self._client_factory is None
                else self._client_factory(self._record_delivery_error)
            )
        return self._client_obj

    def preflight(self) -> None:
        """Require an API key and one cheap authenticated call to succeed (D-043)."""
        if not any(os.environ.get(name, "").strip() for name in API_KEY_ENV_VARS):
            raise ObservabilityError(
                f"no LangSmith API key in {list(API_KEY_ENV_VARS)}", PREFLIGHT_FAILED
            )
        try:
            self._client().has_project(self._project)
        except Exception as error:
            raise ObservabilityError(
                f"LangSmith preflight call failed: {error!r}", PREFLIGHT_FAILED
            ) from error

    @contextmanager
    def span(self, name: str, *, run_type: str, metadata: dict[str, object]) -> Iterator[str]:
        """Open and close one run with redacted metadata; yields the span id."""
        client = self._client()
        run_id = str(uuid.uuid4())
        safe: dict[str, str | int | float | bool] = self._redactor.redact_metadata(metadata)
        safe["environment"] = self._environment
        safe["redaction_policy_version"] = self._redactor.redaction_policy_version
        tags = [f"environment:{self._environment}", f"project:{self._project}"]
        try:
            client.create_run(
                name=self._redactor.redact_text(name),
                inputs={},
                run_type=run_type,
                id=run_id,
                project_name=self._project,
                start_time=datetime.now(UTC),
                extra={"metadata": safe},
                tags=tags,
            )
        except Exception as error:
            raise ObservabilityError(
                f"span {name!r} could not be opened: {error!r}", FLUSH_UNACKNOWLEDGED
            ) from error
        failure: str | None = None
        try:
            yield run_id
        except BaseException as error:
            failure = self._redactor.redact_text(repr(error))
            raise
        finally:
            try:
                client.update_run(run_id, end_time=datetime.now(UTC), error=failure)
            except Exception as close_error:
                if failure is None:
                    raise ObservabilityError(
                        f"span {name!r} could not be closed: {close_error!r}",
                        FLUSH_UNACKNOWLEDGED,
                    ) from close_error

    def flush(self) -> None:
        """Force delivery; raise unless every queued batch was acknowledged."""
        client = self._client()
        try:
            client.flush(timeout=self._flush_timeout_seconds)
        except Exception as error:
            raise ObservabilityError(
                f"LangSmith flush failed: {error!r}", FLUSH_UNACKNOWLEDGED
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
                f"LangSmith rejected a trace batch: {first!r}", FLUSH_UNACKNOWLEDGED
            )
