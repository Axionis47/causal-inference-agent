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
async def test_execute_parks_without_calling_reason(monkeypatch):
    orchestrator = StandardOrchestrator()
    state = _state_at_gate()

    async def _explode(*args, **kwargs):
        raise AssertionError("reason() must not be called when the gate fires")

    monkeypatch.setattr(orchestrator, "reason", _explode)

    result = await orchestrator.execute(state)
    assert result.status == JobStatus.AWAITING_APPROVAL
    assert any(e["event_type"] == "approval_required" for e in result.sse_events)


@pytest.mark.asyncio
async def test_execute_invokes_status_callback_on_park():
    orchestrator = StandardOrchestrator()
    state = _state_at_gate()

    captured: list[JobStatus] = []

    async def cb(s: AnalysisState) -> None:
        captured.append(s.status)

    orchestrator.set_status_callback(cb)

    async def _explode(*args, **kwargs):
        raise AssertionError("reason() must not be called when the gate fires")

    # bind the explosion onto the same instance
    orchestrator.reason = _explode  # type: ignore[assignment]

    await orchestrator.execute(state)
    assert captured == [JobStatus.AWAITING_APPROVAL]


@pytest.mark.asyncio
async def test_execute_skips_gate_when_human_already_approved(monkeypatch):
    """A prior APPROVED decision means we resumed past the gate. The
    orchestrator must call reason() normally instead of parking again."""
    orchestrator = StandardOrchestrator()
    state = _state_at_gate()
    state.human_approval = HumanApproval.approve()

    called = {"reason": 0}

    async def fake_reason(prompt, context):  # noqa: ARG001
        called["reason"] += 1
        # Stop the loop quickly: mark completed so execute returns.
        state.status = JobStatus.COMPLETED
        return {"pending_calls": [], "response": ""}

    monkeypatch.setattr(orchestrator, "reason", fake_reason)

    result = await orchestrator.execute(state)
    assert called["reason"] >= 1
    assert result.status == JobStatus.COMPLETED
