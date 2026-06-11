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
    upcoming = next_stage(state["ctx"].run.current_state)
    if upcoming is None or upcoming == AnalysisStage.S12_JOB_COMPLETE:
        return END
    return upcoming.value


def _make_dispatch(agents: dict[AnalysisStage, AnalysisAgent]):
    async def dispatch(state: SpineState) -> dict[str, Any]:
        # A node already decided the run's fate; dispatch must not march on.
        if state["outcome"] != "running":
            return {"outcome": state["outcome"]}
        ctx = state["ctx"]
        run = ctx.run
        if run.status in (RunStatus.FAILED, RunStatus.CANCELLED):
            return {"outcome": "failed"}

        upcoming = next_stage(run.current_state)
        if upcoming == AnalysisStage.S6_USER_CONFIRMED_OR_AUTO_APPROVED:
            verdict = await _pass_plan_gate(ctx)
            if verdict != "running":
                return {"outcome": verdict}
            upcoming = next_stage(run.current_state)

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
            ctx.emit("analysis_failed", {"headline": message, "error": message})
            return {"outcome": "frontier"}
        return {"outcome": "running"}

    return dispatch


async def _pass_plan_gate(ctx: AgentCtx) -> str:
    """S6 is orchestrator-owned: auto-approve advances, anything else parks.

    A user confirmation records the S6 transition through the resume path
    before relaunch, so a run sitting here with NEEDS_USER parks again
    (defensive: a crash between confirm and relaunch loses nothing).
    """
    from src.analysis_v2.core import GateResult
    from src.analysis_v2.spec import PlanGateStatus

    run = ctx.run
    critique = run.plan_critique
    if critique is None:
        run.mark_failed("plan gate reached without a plan critique")
        await save_run(run)
        return "failed"
    if critique.status == PlanGateStatus.PASS_AUTO_APPROVED:
        run.record_transition(
            to_state=AnalysisStage.S6_USER_CONFIRMED_OR_AUTO_APPROVED,
            agent_name=None,
            gate_result=GateResult.advance(),
        )
        await save_run(run)
        ctx.emit(
            "analysis_stage_started",
            {
                "stage": AnalysisStage.S6_USER_CONFIRMED_OR_AUTO_APPROVED.value,
                "agent": "orchestrator",
                "headline": "plan auto-approved",
            },
        )
        return "running"
    if critique.status == PlanGateStatus.NEEDS_USER_CONFIRMATION:
        run.status = RunStatus.WAITING_FOR_USER
        await save_run(run)
        return "parked"
    run.mark_failed(
        "plan gate cannot pass: " + "; ".join(critique.missing_required or ["unknown"])
    )
    await save_run(run)
    return "failed"


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
