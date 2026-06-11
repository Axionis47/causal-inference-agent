"""The LangGraph spine: a dispatch router plus one node per agent stage.

Control flow is deterministic: dispatch reads run.current_state and routes
to the next stage's node; every agent node returns to dispatch. The gate
result decides what dispatch sees next (advance, park for the user, fail).
Stages without a registered agent are the build frontier: the run stops
there with an explicit failure instead of pretending completion. Reaching
S11 with everything green completes the job (S12).
"""
from __future__ import annotations

from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from src.analysis_v2.agents.base import AgentCtx, AnalysisAgent
from src.analysis_v2.core import (
    AnalysisStage,
    GateStatus,
    RunStatus,
    next_stage,
)
from src.analysis_v2.persistence import save_run

from .execute import run_stage

logger = structlog.get_logger(__name__)

_DISPATCH = "dispatch"


class SpineState(TypedDict):
    ctx: AgentCtx
    outcome: str  # running | parked | failed | complete | frontier


def _route(state: SpineState) -> str:
    """Conditional edge from dispatch: the next node name or END."""
    if state["outcome"] != "running":
        return END
    run = state["ctx"].run
    upcoming = next_stage(run.current_state)
    # S6 is orchestrator-owned: until the plan gate exists it auto-advances
    # only when the plan critic already approved (M6 wires the real gate).
    while upcoming is not None and upcoming in (
        AnalysisStage.S6_USER_CONFIRMED_OR_AUTO_APPROVED,
    ):
        upcoming = next_stage(upcoming)
    if upcoming is None or upcoming == AnalysisStage.S12_JOB_COMPLETE:
        return END
    return upcoming.value


def _make_dispatch(agents: dict[AnalysisStage, AnalysisAgent]):
    async def dispatch(state: SpineState) -> dict[str, Any]:
        # A node already decided the run's fate; dispatch must not march on.
        if state["outcome"] != "running":
            return {"outcome": state["outcome"]}
        run = state["ctx"].run
        if run.status in (RunStatus.FAILED, RunStatus.CANCELLED):
            return {"outcome": "failed"}
        upcoming = next_stage(run.current_state)
        while upcoming == AnalysisStage.S6_USER_CONFIRMED_OR_AUTO_APPROVED:
            upcoming = next_stage(upcoming)
        if upcoming is None or upcoming == AnalysisStage.S12_JOB_COMPLETE:
            return {"outcome": "complete"}
        if upcoming not in agents:
            message = (
                f"analysis spine ends at {run.current_state.value}: "
                f"stage {upcoming.value} has no implementation yet"
            )
            logger.warning("spine_frontier_reached", job_id=run.job_id, stage=upcoming.value)
            run.mark_failed(message)
            await save_run(run)
            state["ctx"].emit(
                "analysis_failed", {"headline": message, "error": message}
            )
            return {"outcome": "frontier"}
        return {"outcome": "running"}

    return dispatch


def _make_node(agent: AnalysisAgent):
    async def node(state: SpineState) -> dict[str, Any]:
        gate = await run_stage(agent, state["ctx"])
        if gate == GateStatus.FAIL:
            return {"outcome": "failed"}
        if gate == GateStatus.NEEDS_USER:
            run = state["ctx"].run
            run.status = RunStatus.WAITING_FOR_USER
            await save_run(run)
            return {"outcome": "parked"}
        return {"outcome": "running"}

    return node


def build_spine(agents: dict[AnalysisStage, AnalysisAgent]):
    graph: StateGraph = StateGraph(SpineState)
    graph.add_node(_DISPATCH, _make_dispatch(agents))
    targets = {END: END}
    for stage, agent in agents.items():
        graph.add_node(stage.value, _make_node(agent))
        graph.add_edge(stage.value, _DISPATCH)
        targets[stage.value] = stage.value
    graph.add_conditional_edges(_DISPATCH, _route, targets)
    graph.set_entry_point(_DISPATCH)
    return graph.compile()


async def run_spine(ctx: AgentCtx, agents: dict[AnalysisStage, AnalysisAgent]) -> str:
    """Drive the spine to its stopping point; returns the final outcome."""
    spine = build_spine(agents)
    state: SpineState = {"ctx": ctx, "outcome": "running"}
    final = await spine.ainvoke(state, config={"recursion_limit": 100})
    return final["outcome"]
