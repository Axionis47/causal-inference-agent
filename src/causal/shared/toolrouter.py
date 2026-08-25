"""Allowlist-enforcing tool router shared by every stage harness (SC §5.4; D-049)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from causal.shared.envelope import AgentTaskEnvelopeV1
from causal.shared.events import EventEmitter, OperationalEventV1, Severity, Stage, build_event

__all__ = ["ToolError", "ToolHandler", "ToolResult", "ToolRouter", "tool_allowlists"]

TOOL_DENIED: Final = "tool_denied"
TOOL_FAILED: Final = "tool_failed"
UNKNOWN_TOOL: Final = "unknown_tool"
TOOL_EVAL_IDS: Final = ("EV-SYS-002",)

ToolHandler = Callable[[AgentTaskEnvelopeV1, Mapping[str, Any]], dict[str, Any]]


class ToolError(ValueError):
    """A tool call was refused or failed. `code` is a stable contract value."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ToolResult:
    """One completed tool call: bounded scalars and lists, never rows or documents."""

    tool_id: str
    status: str
    result: dict[str, Any]


def tool_allowlists(
    document: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], dict[str, bool]]:
    """Split a parsed `design-tools.v1` document into allowlist and registration maps."""
    rows = next((value for value in document.values() if isinstance(value, list)), [])
    return (
        {str(r["tool_id"]): tuple(str(k) for k in r["allowed_task_kinds"]) for r in rows},
        {str(r["tool_id"]): bool(r["registered"]) for r in rows},
    )


class ToolRouter:
    """Enforcement order: listed, registered, task-kind allowlisted, envelope, handler."""

    def __init__(
        self,
        allowlists: Mapping[str, tuple[str, ...]],
        registered: Mapping[str, bool],
        handlers: Mapping[str, ToolHandler],
        emitter: EventEmitter,
        clock: Callable[[], datetime],
        component_version: str = "design-harness.v1",
        component_id: str = "design-harness",
        stage: Stage = Stage.DESIGN,
        tool_registry_version: str = "design-tools.v1",
    ) -> None:
        self._allowlists = dict(allowlists)
        self._registered = dict(registered)
        self._handlers = dict(handlers)
        self._emitter = emitter
        self._clock = clock
        self._component_version = component_version
        self._component_id = component_id
        self._stage = stage
        self._tool_registry_version = tool_registry_version
        self._counts: dict[str, int] = {}

    def _event(
        self, envelope: AgentTaskEnvelopeV1, name: str, tool_id: str, **overrides: object
    ) -> OperationalEventV1:
        count = self._counts.get(envelope.envelope_id, 0) + 1
        self._counts[envelope.envelope_id] = count
        return build_event(
            occurred_at_utc=self._clock(), event_name=name,
            event_id=f"evt:{envelope.envelope_id}:tool:{count}",
            analysis_id=envelope.analysis_id, stage=self._stage,
            stage_run_id=envelope.stage_run_id, task_id=envelope.task_id,
            attempt_id=envelope.attempt_id, component_id=self._component_id,
            component_version=self._component_version,
            versions={"tool": self._tool_registry_version}, required_eval_ids=TOOL_EVAL_IDS,
            safe_dimensions={"tool_id": tool_id, "task_kind": envelope.task_kind},
            **overrides,
        )

    def _deny(
        self, envelope: AgentTaskEnvelopeV1, tool_id: str, code: str, reason: str
    ) -> ToolError:
        self._emitter.emit(self._event(
            envelope, "tool.denied", tool_id, severity=Severity.ERROR, status="denied",
            error_code=code, retryable=False))
        return ToolError(f"tool {tool_id!r} denied: {reason}", code)

    def call(
        self, envelope: AgentTaskEnvelopeV1, tool_id: str, arguments: Mapping[str, Any]
    ) -> ToolResult:
        """Run one allowlisted tool; every refusal emits `tool.denied` and calls no handler."""
        allowed = self._allowlists.get(tool_id)
        if allowed is None:
            raise self._deny(envelope, tool_id, UNKNOWN_TOOL, "not listed in the registry")
        if not self._registered.get(tool_id, False):
            raise self._deny(envelope, tool_id, TOOL_DENIED, "not registered in this version")
        if envelope.task_kind not in allowed:
            raise self._deny(
                envelope, tool_id, TOOL_DENIED,
                f"task kind {envelope.task_kind!r} is not allowlisted")
        if tool_id not in envelope.allowed_tool_ids:
            raise self._deny(envelope, tool_id, TOOL_DENIED, "absent from the envelope")
        handler = self._handlers.get(tool_id)
        if handler is None:
            raise self._deny(envelope, tool_id, UNKNOWN_TOOL, "no handler is bound")
        self._emitter.emit(self._event(envelope, "tool.started", tool_id))
        try:
            result = handler(envelope, arguments)
        except Exception as error:  # any handler failure is exactly one tool failure
            self._emitter.emit(self._event(
                envelope, "tool.failed", tool_id, severity=Severity.ERROR, status="failed",
                error_code=TOOL_FAILED, retryable=False,
                exception_class=type(error).__name__))
            raise ToolError(f"tool {tool_id!r} failed: {error}", TOOL_FAILED) from error
        self._emitter.emit(self._event(envelope, "tool.completed", tool_id, status="completed"))
        return ToolResult(tool_id=tool_id, status="completed", result=result)
