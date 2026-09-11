"""OperationalEventV1 and the NDJSON event emitter (SYSTEM-CONTRACT §10.1; D-013..D-016)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from causal.shared.canonical import canonical_bytes
from causal.shared.contracts import ArtifactRef, Identity, Sha256Hex, UtcTimestamp

__all__ = [
    "EVENT_NAMES_V1",
    "EventEmitter",
    "EventEmitterError",
    "OperationalEventV1",
    "Severity",
    "Stage",
    "build_event",
]

UNREGISTERED_EVENT_NAME: Final = "unregistered_event_name"
MISSING_REQUIRED_EVAL_IDS: Final = "missing_required_eval_ids"

EVENT_NAMES_V1: Final[frozenset[str]] = frozenset(
    {
        "stage.started", "stage.completed", "stage.failed",
        "task.started", "task.completed", "task.failed",
        "artifact.committed", "artifact.validation_failed",
        "agent.started", "agent.schema_failed", "agent.correction_requested",
        "agent.diagnostic_requested", "agent.design_revised", "agent.escalated",
        "diagnostic.completed",
        "tool.started", "tool.completed", "tool.denied", "tool.failed",
        "retry.scheduled", "retry.exhausted",
        "user_interrupt.created", "user_interrupt.resumed",
        "handoff.accepted", "handoff.rejected",
        "observability.delivery_failed",
        "blocker.raised",
        "evaluation.run_started", "evaluation.case_started",
        "evaluation.case_completed", "evaluation.run_completed",
        "evaluation.release_blocked",
    }
)

_VERSION_KEYS: Final = frozenset(
    {"model", "prompt", "tool", "registry", "schema", "validator", "compiler", "renderer"}
)
_TOKEN_USAGE_KEYS: Final = frozenset({"input", "output", "thinking", "total"})
_EVAL_REQUIRED_PREFIXES: Final = ("task.", "agent.", "diagnostic.", "tool.", "handoff.")


class Severity(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Stage(StrEnum):
    INTAKE = "intake"
    DESIGN = "design"
    PREPARATION = "preparation"
    ESTIMATION = "estimation"
    PRESENTATION = "presentation"
    SYSTEM = "system"


EventName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")]


class OperationalEventV1(BaseModel):
    """One machine-readable operational event, emitted as a single NDJSON line."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["operational-event.v1"]
    occurred_at_utc: UtcTimestamp
    severity: Severity
    event_name: EventName
    event_id: Identity
    parent_event_id: Identity | None
    analysis_id: Identity
    stage: Stage
    stage_run_id: Identity
    graph_thread_id: Identity | None
    task_id: Identity | None
    attempt_id: Identity | None
    attempt_number: Annotated[int, Field(ge=1)] | None
    component_id: Identity
    component_version: Identity
    versions: dict[str, Identity]
    status: Identity | None
    error_code: Identity | None
    retryable: bool | None
    duration_ms: Annotated[float, Field(ge=0)] | None
    token_usage: dict[str, Annotated[int, Field(ge=0)]]
    cost: Annotated[float, Field(ge=0)] | None
    artifact_refs: tuple[ArtifactRef, ...]
    required_eval_ids: tuple[Identity, ...]
    evaluation_run_id: Identity | None
    evaluation_case_id: Identity | None
    evaluation_fixture_hash: Sha256Hex | None
    evaluator_version: Identity | None
    evaluation_gate_status: Identity | None
    exception_class: Identity | None
    exception_fingerprint: Identity | None
    safe_dimensions: dict[str, str | int | float | bool]

    @field_validator("versions")
    @classmethod
    def _version_keys_registered(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = set(value) - _VERSION_KEYS
        if unknown:
            raise ValueError(f"unregistered version keys: {sorted(unknown)}")
        return value

    @field_validator("token_usage")
    @classmethod
    def _token_usage_keys_registered(cls, value: dict[str, int]) -> dict[str, int]:
        unknown = set(value) - _TOKEN_USAGE_KEYS
        if unknown:
            raise ValueError(f"unregistered token_usage keys: {sorted(unknown)}")
        return value


_EVENT_DEFAULTS: Final[dict[str, object]] = {
    "schema_version": "operational-event.v1", "severity": Severity.INFO,
    "parent_event_id": None, "graph_thread_id": None, "task_id": None,
    "attempt_id": None, "attempt_number": None, "versions": {}, "status": None,
    "error_code": None, "retryable": None, "duration_ms": None, "token_usage": {},
    "cost": None, "artifact_refs": (), "required_eval_ids": (),
    "evaluation_run_id": None, "evaluation_case_id": None,
    "evaluation_fixture_hash": None, "evaluator_version": None,
    "evaluation_gate_status": None, "exception_class": None,
    "exception_fingerprint": None, "safe_dimensions": {},
}


def build_event(**fields: object) -> OperationalEventV1:
    """An event with every optional field defaulted; callers override as needed."""
    return OperationalEventV1(**{**_EVENT_DEFAULTS, **fields})  # type: ignore[arg-type]


class EventEmitterError(ValueError):
    """An event cannot be emitted. `code` is a stable contract value."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class EventEmitter:
    """Writes one canonical NDJSON line per event to the configured log sink."""

    def __init__(self, sink: TextIO, registered_names: frozenset[str] = EVENT_NAMES_V1) -> None:
        self._sink = sink
        self._registered_names = registered_names

    def emit(self, event: OperationalEventV1) -> None:
        if event.event_name not in self._registered_names:
            raise EventEmitterError(
                f"event name not registered: {event.event_name!r}", UNREGISTERED_EVENT_NAME
            )
        if event.event_name.startswith(_EVAL_REQUIRED_PREFIXES) and not event.required_eval_ids:
            raise EventEmitterError(
                f"{event.event_name!r} requires non-empty required_eval_ids",
                MISSING_REQUIRED_EVAL_IDS,
            )
        line = canonical_bytes(event.model_dump(mode="json")).decode("utf-8")
        self._sink.write(line + "\n")
        self._sink.flush()
