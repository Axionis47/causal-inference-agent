"""ReActOrchestrator must park, not dispatch, when the approval gate fires.

The ReAct loop calls `is_task_complete` after every tool execution. The
gate makes AWAITING_APPROVAL run-terminal so the loop ends cleanly; the
`_dispatch_agent` tool refuses dispatch and parks instead, so a paused
job cannot be advanced even by an LLM that doesn't know about the gate.
"""
from __future__ import annotations

import pytest

from src.analysis.agents.base import DataProfile
from src.analysis.agents.base.react_agent import ToolResultStatus
from src.analysis.agents.base.state import AnalysisState, DatasetInfo, JobStatus
from src.analysis.orchestrator.react.agent import ReActOrchestrator
from src.domain.approval import HumanApproval


def _state_at_gate() -> AnalysisState:
    return AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(url="kaggle.com/x"),
        treatment_variable="t",
        outcome_variable="y",
        data_profile=DataProfile(
            n_samples=614,
            n_features=11,
            feature_names=["t", "y", "x1"],
            feature_types={"t": "binary"},
            missing_values={"t": 0},
            numeric_stats={},
            categorical_stats={},
            treatment_candidates=["t"],
            outcome_candidates=["y"],
        ),
    )


# --- is_task_complete -------------------------------------------------------


@pytest.mark.asyncio
async def test_is_task_complete_treats_awaiting_approval_as_terminal():
    orchestrator = ReActOrchestrator()
    state = _state_at_gate()
    state.status = JobStatus.AWAITING_APPROVAL
    assert await orchestrator.is_task_complete(state) is True


@pytest.mark.asyncio
async def test_is_task_complete_still_recognises_completed_and_failed():
    orchestrator = ReActOrchestrator()
    state = _state_at_gate()
    state.status = JobStatus.COMPLETED
    assert await orchestrator.is_task_complete(state) is True
    state.status = JobStatus.FAILED
    assert await orchestrator.is_task_complete(state) is True


@pytest.mark.asyncio
async def test_is_task_complete_returns_false_for_in_progress_states():
    orchestrator = ReActOrchestrator()
    state = _state_at_gate()
    state.status = JobStatus.DISCOVERING_CAUSAL
    assert await orchestrator.is_task_complete(state) is False


# --- _dispatch_agent gate behaviour ----------------------------------------


@pytest.mark.asyncio
async def test_dispatch_parks_instead_of_running_when_gate_is_open():
    orchestrator = ReActOrchestrator()
    state = _state_at_gate()

    # No specialist registered: if the gate check didn't fire, the lookup
    # would error. The gate check must be ahead of the specialist lookup.
    result = await orchestrator._dispatch_agent(state, agent_name="effect_estimator")
    assert result.status == ToolResultStatus.SUCCESS
    assert result.output["paused_for_approval"] is True
    assert state.status == JobStatus.AWAITING_APPROVAL
    assert any(e["event_type"] == "approval_required" for e in state.sse_events)


@pytest.mark.asyncio
async def test_dispatch_invokes_status_callback_on_park():
    orchestrator = ReActOrchestrator()
    state = _state_at_gate()

    captured: list[JobStatus] = []

    async def cb(s: AnalysisState) -> None:
        captured.append(s.status)

    orchestrator.set_status_callback(cb)
    await orchestrator._dispatch_agent(state, agent_name="effect_estimator")
    assert captured == [JobStatus.AWAITING_APPROVAL]


@pytest.mark.asyncio
async def test_dispatch_parks_even_for_non_effect_estimator_when_gate_open():
    """Defence in depth: any dispatch attempt while the gate is open parks.
    Otherwise an LLM could try to call critique or dag_expert again and
    skip the human review entirely."""
    orchestrator = ReActOrchestrator()
    state = _state_at_gate()

    result = await orchestrator._dispatch_agent(state, agent_name="critique")
    assert result.output["paused_for_approval"] is True
    assert "critique" in result.output["message"]


@pytest.mark.asyncio
async def test_dispatch_skips_gate_after_human_approved():
    """A prior APPROVED decision means we resumed past the gate. The
    dispatch should proceed to the specialist lookup (and fail there
    because no specialist is registered — that's the marker that the
    gate did not fire)."""
    orchestrator = ReActOrchestrator()
    state = _state_at_gate()
    state.human_approval = HumanApproval.approve()

    result = await orchestrator._dispatch_agent(state, agent_name="effect_estimator")
    # Past the gate: now we hit the missing-specialist error path.
    assert result.status == ToolResultStatus.ERROR
    assert "not registered" in (result.error or "")
