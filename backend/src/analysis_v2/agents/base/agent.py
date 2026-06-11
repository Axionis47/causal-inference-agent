"""The agent contract every analysis worker implements.

An agent is either a deterministic tool-wrapped node or a local ReAct
loop; either way it returns an AgentResult and never mutates run-state
slots itself. The orchestrator validates the typed output, calls the
agent's `commit` to place it in its one slot, and records the transition.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.analysis_v2.core import (
    AgentFailure,
    AnalysisRunState,
    AnalysisStage,
    GateResult,
    TokenUsage,
    ToolCallRecord,
)

from .context import AgentCtx


@dataclass
class AgentResult:
    """What one dispatch hands back to the orchestrator."""

    gate: GateResult
    output: Any = None  # the typed payload `commit` knows how to place
    public_summary: str = ""
    warnings: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    failure: AgentFailure | None = None


class AnalysisAgent(ABC):
    """One worker on the spine. Subclasses set `name` and `stage`."""

    name: str
    stage: AnalysisStage

    @abstractmethod
    async def execute(self, ctx: AgentCtx) -> AgentResult:
        """Do the stage's work. Read through ctx; never mutate run slots."""

    @abstractmethod
    def commit(self, run: AnalysisRunState, output: Any) -> None:
        """Place the validated output into this agent's slot on the run."""
