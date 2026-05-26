"""Orchestrator protocol shared by all orchestrator implementations.

The orchestrator coordinates specialist agents to run a causal-inference
analysis end to end. JobManager depends on this Protocol rather than a
concrete class, so swapping between standard and react orchestration
is a configuration choice, not a code change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.analysis.agents.base.agent import BaseAgent
    from src.analysis.agents.base.state import AnalysisState


@runtime_checkable
class Orchestrator(Protocol):
    """Contract every orchestrator implementation must satisfy.

    Implementations today:
        StandardOrchestrator (standard/agent.py): LLM-driven dispatch
            with a default workflow suggested in the system prompt.
        ReActOrchestrator (react/agent.py): Fully autonomous ReAct loop
            with no fixed workflow.

    Both inherit execute_with_tracing from BaseAgent / ReActAgent, which
    is the shared tracing and logging wrapper. JobManager invokes that
    wrapper rather than execute directly so AgentTrace recording and
    structlog emission happen uniformly across orchestrators.
    """

    def register_specialist(self, name: str, agent: BaseAgent) -> None:
        """Register a specialist the orchestrator can dispatch to."""
        ...

    async def execute_with_tracing(self, state: AnalysisState) -> AnalysisState:
        """Run the orchestration loop with tracing and logging applied.

        Returns the updated state. In-pipeline failures mark the state
        FAILED rather than raising.
        """
        ...
