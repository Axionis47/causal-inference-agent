"""Per-agent execution records: what one dispatch did, cost, and showed.

One AgentRun per agent dispatch. The frontend tile is rendered straight
from this record: status, public reasoning summary, tool calls, tokens,
cost, warnings, artifacts, gate result. `public_summary` is a concise
human-readable account; private chain-of-thought is never stored.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field

from .failure import AgentFailure
from .gates import GateResult
from .stages import AnalysisStage


class AgentRunStatus(StrEnum):
    WAITING = "waiting"
    RUNNING = "running"
    PASSED = "passed"
    WARNING = "warning"  # passed, but with soft warnings worth showing
    FAILED = "failed"


class TokenUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    def add(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class ToolCallRecord(BaseModel):
    name: str = Field(min_length=1)
    args_summary: str = Field(default="", max_length=500)
    ok: bool = True
    error: str | None = Field(default=None, max_length=500)
    duration_ms: int | None = Field(default=None, ge=0)


class AgentRun(BaseModel):
    agent: str = Field(min_length=1)
    stage: AnalysisStage
    status: AgentRunStatus = AgentRunStatus.WAITING
    attempt: int = Field(default=1, ge=1)
    current_step: str | None = Field(default=None, max_length=200)
    public_summary: str | None = Field(default=None, max_length=4000)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = Field(default=0.0, ge=0.0)
    warnings: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    gate_result: GateResult | None = None
    failure: AgentFailure | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def start(self) -> None:
        self.status = AgentRunStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def finish(self, status: AgentRunStatus) -> None:
        if status in (AgentRunStatus.WAITING, AgentRunStatus.RUNNING):
            raise ValueError("finish requires a terminal agent-run status")
        self.status = status
        self.finished_at = datetime.now(timezone.utc)

    @property
    def elapsed_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()
