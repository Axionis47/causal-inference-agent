"""Per-job budget tracking with three-level caps.

BudgetTracker wraps a BudgetSpec from domain.job and counts every
cost-incurring event in a job:

- agent_loops: one per orchestrator dispatch of a specialist
- tool_calls: per-agent and total
- llm_tokens: cumulative across the job
- wall clock: seconds since the tracker was created

A tripped cap raises BudgetExceeded. The orchestrator catches it,
marks the affected slot as partial, and finalises with what it has.
The job is never crashed by a budget trip; the snapshot is recorded
in the trace so the user sees which cap stopped which agent.

The clock is injectable so tests can advance time without sleeping.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.domain.job import BudgetSpec

BudgetCap = Literal[
    "max_agent_loops",
    "max_tool_calls_per_agent",
    "max_total_tool_calls",
    "max_llm_tokens",
    "wall_clock_seconds",
]


class BudgetExceeded(Exception):
    """Raised when a cap in BudgetSpec is hit.

    The orchestrator catches this and finalises gracefully. The job is
    not crashed; the agent's produced slot is marked partial.
    """

    def __init__(self, cap: BudgetCap, current: int, limit: int) -> None:
        super().__init__(f"Budget exceeded: {cap} reached {current}/{limit}")
        self.cap = cap
        self.current = current
        self.limit = limit


class BudgetSnapshot(BaseModel):
    """Serializable view of a tracker, for traces and final state."""

    model_config = ConfigDict(extra="forbid")

    agent_loops: int
    total_tool_calls: int
    tool_calls_by_agent: dict[str, int]
    llm_tokens: int
    elapsed_seconds: float


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BudgetTracker:
    """Mutable per-job counter, checked at every cost-incurring event.

    Caps in BudgetSpec are inclusive: max_agent_loops=20 allows 20 loops
    and rejects the 21st. Same convention for the other caps.
    """

    def __init__(
        self,
        spec: BudgetSpec,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._spec = spec
        self._clock = clock
        self._started_at = clock()
        self._agent_loops = 0
        self._tool_calls_by_agent: dict[str, int] = {}
        self._llm_tokens = 0

    @property
    def spec(self) -> BudgetSpec:
        return self._spec

    @property
    def agent_loops(self) -> int:
        return self._agent_loops

    @property
    def total_tool_calls(self) -> int:
        return sum(self._tool_calls_by_agent.values())

    @property
    def llm_tokens(self) -> int:
        return self._llm_tokens

    def tool_calls_for(self, agent: str) -> int:
        return self._tool_calls_by_agent.get(agent, 0)

    def elapsed_seconds(self) -> float:
        return (self._clock() - self._started_at).total_seconds()

    def record_agent_loop(self) -> None:
        self._agent_loops += 1
        if self._agent_loops > self._spec.max_agent_loops:
            raise BudgetExceeded(
                "max_agent_loops", self._agent_loops, self._spec.max_agent_loops
            )
        self._check_wall_clock()

    def record_tool_call(self, agent: str) -> None:
        self._tool_calls_by_agent[agent] = self._tool_calls_by_agent.get(agent, 0) + 1
        per_agent = self._tool_calls_by_agent[agent]
        if per_agent > self._spec.max_tool_calls_per_agent:
            raise BudgetExceeded(
                "max_tool_calls_per_agent",
                per_agent,
                self._spec.max_tool_calls_per_agent,
            )
        total = self.total_tool_calls
        if total > self._spec.max_total_tool_calls:
            raise BudgetExceeded(
                "max_total_tool_calls", total, self._spec.max_total_tool_calls
            )
        self._check_wall_clock()

    def record_llm_tokens(self, n: int) -> None:
        if n < 0:
            raise ValueError("LLM token count must be non-negative")
        self._llm_tokens += n
        if self._llm_tokens > self._spec.max_llm_tokens:
            raise BudgetExceeded(
                "max_llm_tokens", self._llm_tokens, self._spec.max_llm_tokens
            )
        self._check_wall_clock()

    def _check_wall_clock(self) -> None:
        elapsed = self.elapsed_seconds()
        if elapsed > self._spec.wall_clock_seconds:
            raise BudgetExceeded(
                "wall_clock_seconds", int(elapsed), self._spec.wall_clock_seconds
            )

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            agent_loops=self._agent_loops,
            total_tool_calls=self.total_tool_calls,
            tool_calls_by_agent=dict(self._tool_calls_by_agent),
            llm_tokens=self._llm_tokens,
            elapsed_seconds=self.elapsed_seconds(),
        )
