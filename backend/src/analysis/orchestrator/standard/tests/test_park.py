"""StandardOrchestrator must park, not reason, when the approval gate fires.

The orchestrator's decision loop checks the gate predicate at the top of
every iteration. When it fires (once the data is profiled) we must exit
before `reason()` is called — otherwise the LLM will dispatch analysis in
violation of the data-review contract. These tests pin that boundary.
"""
from __future__ import annotations

import pytest

from src.analysis.agents.base import DataProfile
from src.analysis.agents.base.state import AnalysisState, DatasetInfo, JobStatus
from src.analysis.orchestrator.standard.agent import StandardOrchestrator
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


@pytest.mark.asyncio
async def test_execute_parks_at_the_data_gate_before_any_analysis():
    """A profiled, not-yet-approved state parks for review before the spine
    dispatches any analysis agent."""
    orchestrator = StandardOrchestrator()
    state = _state_at_gate()

    result = await orchestrator.execute(state)

    assert result.status == JobStatus.AWAITING_APPROVAL
    assert any(e["event_type"] == "approval_required" for e in result.sse_events)
    # Parking happened before any dispatch: no specialists are registered, so a
    # dispatch would have failed the job instead of parking it cleanly.
    assert result.error_message is None


@pytest.mark.asyncio
async def test_execute_invokes_status_callback_on_park():
    orchestrator = StandardOrchestrator()
    state = _state_at_gate()

    captured: list[JobStatus] = []

    async def cb(s: AnalysisState) -> None:
        captured.append(s.status)

    orchestrator.set_status_callback(cb)

    await orchestrator.execute(state)
    assert captured == [JobStatus.AWAITING_APPROVAL]


@pytest.mark.asyncio
async def test_execute_does_not_park_again_when_already_approved():
    """A prior APPROVED decision means we resumed past the gate. The
    orchestrator must proceed into the spine, not park again."""
    orchestrator = StandardOrchestrator()
    state = _state_at_gate()
    state.human_approval = HumanApproval.approve()

    result = await orchestrator.execute(state)

    # It advanced into the spine (and only fails here because no specialists are
    # registered in this test), rather than parking at the data gate again.
    assert result.status != JobStatus.AWAITING_APPROVAL
