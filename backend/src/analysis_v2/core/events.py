"""Immutable state-transition events: the append-only audit log of a run.

One StateEvent per committed transition. Events are frozen; the run state
appends and never edits. This log is how a reopened job replays what
happened and how drift between slots and history is caught.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from .agent_run import TokenUsage
from .gates import GateResult
from .stages import AnalysisStage


class StateEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    sequence: int = Field(ge=0)
    from_state: AnalysisStage
    to_state: AnalysisStage
    agent_name: str | None = None
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    gate_result: GateResult | None = None
    warnings: list[str] = Field(default_factory=list)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
