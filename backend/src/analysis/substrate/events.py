"""SSE event bus and decision audit log.

Two bounded buffers an agent writes into during a job, sharing one
shape: FIFO trim from the front when full, callable-style emit at the
boundary, typed model under the hood.

- SSEEventBus + emit_event(bus, event_type, data): the live-update
  stream the front-end Data panel and JobPage consume. Event names are
  a wire contract (see CLAUDE.md section 5) and must be preserved
  byte-for-byte; the bus only owns capacity and ordering.

- DecisionLog + record_decision(log, **fields): the decision audit
  trail. Read by the notebook's decisions section and persisted with
  the job. Each entry carries the chosen path plus any alternatives
  the agent considered, so the final report can show "why this and
  not that."

Both buffers are mutated only by whoever holds the job state (single
event loop, no locks). Cap defaults match the legacy state values
(100 / 100); the orchestrator builds each with settings-driven caps.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SSEEvent(BaseModel):
    """One live-update event streamed to the front-end.

    The wire shape is `{event_type, data, timestamp}` where timestamp
    is an ISO 8601 string. We carry a datetime here and let callers
    serialise on emit; the legacy shape comes back via model_dump.
    """

    model_config = ConfigDict(extra="forbid")

    event_type: str
    data: dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SSEEventBus:
    """Bounded FIFO buffer for SSE events.

    The legacy `state.sse_events` was a plain list[dict] sliced by
    offset for polling (JobManager.get_sse_events). view_after preserves
    that contract so JobManager's migration in S6 is a drop-in.
    """

    def __init__(self, max_events: int = 100) -> None:
        if max_events < 1:
            raise ValueError("max_events must be >= 1")
        self._max = max_events
        self._events: list[SSEEvent] = []

    def __len__(self) -> int:
        return len(self._events)

    @property
    def max_events(self) -> int:
        return self._max

    def append(self, event: SSEEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._max:
            self._events = self._events[-self._max :]

    def view(self) -> list[SSEEvent]:
        return list(self._events)

    def view_after(self, after_index: int) -> list[SSEEvent]:
        """Events at positions >= after_index. Matches legacy slice."""
        return self._events[after_index:]


def emit_event(
    bus: SSEEventBus,
    event_type: str,
    data: dict[str, Any],
) -> SSEEvent:
    """Build an SSEEvent and append it to the bus. Returns the event."""
    event = SSEEvent(event_type=event_type, data=data)
    bus.append(event)
    return event


class AnalysisDecision(BaseModel):
    """One entry in the decision audit trail.

    Relocated verbatim from agents/base/state.py. The notebook section,
    Firestore serializer, and react-orchestrator boundary merge all
    read this shape; do not change field names or types.
    """

    model_config = ConfigDict(extra="forbid")

    agent: str
    decision_type: str
    choice: str
    reason: str
    alternatives: list[dict[str, str]] = Field(default_factory=list)
    timestamp: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class DecisionLog:
    """Bounded FIFO buffer for AnalysisDecision entries."""

    def __init__(self, max_decisions: int = 100) -> None:
        if max_decisions < 1:
            raise ValueError("max_decisions must be >= 1")
        self._max = max_decisions
        self._decisions: list[AnalysisDecision] = []

    def __len__(self) -> int:
        return len(self._decisions)

    @property
    def max_decisions(self) -> int:
        return self._max

    def append(self, decision: AnalysisDecision) -> None:
        self._decisions.append(decision)
        if len(self._decisions) > self._max:
            self._decisions = self._decisions[-self._max :]

    def view(self) -> list[AnalysisDecision]:
        return list(self._decisions)


def record_decision(
    log: DecisionLog,
    *,
    agent: str,
    decision_type: str,
    choice: str,
    reason: str,
    alternatives: list[dict[str, str]] | None = None,
) -> AnalysisDecision:
    """Build an AnalysisDecision and append it to the log."""
    decision = AnalysisDecision(
        agent=agent,
        decision_type=decision_type,
        choice=choice,
        reason=reason,
        alternatives=alternatives or [],
    )
    log.append(decision)
    return decision
