"""JobManager.resume_from_approval end-to-end against real storage.

Drives the full flow: a parked state lands on disk via save_parked_state,
the human submits an approval with DAG edits + appended context, the
manager (a) merges the edits into refined_dag in place, (b) appends the
context to dataset_info.user_provided_context, (c) attaches the
HumanApproval record, (d) deletes the parked record, (e) respawns a
fresh task whose state carries all the above.

The respawn is captured by patching asyncio.create_task so the test does
not actually run the orchestrator — what's pinned is the *contract* that
the post-approval state matches what effect_estimator will read on next
dispatch (proposed_dag.adjustment_set + user_provided_context).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis.agents.base.state import (
    AnalysisState,
    DatasetInfo,
    JobStatus,
)
from src.analysis.agents.causal_discovery.output import CausalDAG, CausalEdge
from src.analysis.agents.eda.output import EDAResult
from src.domain.approval import (
    ApprovalDecision,
    DagEdit,
    HumanApproval,
)


@pytest.fixture()
def manager_against_real_local_storage(tmp_path, monkeypatch):
    """JobManager wired against a real LocalStorageClient at tmp_path.

    Only the orchestrator-build + the respawn task creation are stubbed —
    everything between the API and the storage is real, so the test
    exercises the edit-merge and persistence contract end to end.
    """
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))

    from src.config import settings as settings_mod
    settings_mod.get_settings.cache_clear()
    fake_settings = settings_mod.get_settings()

    with (
        patch("src.jobs.manager.standard_pkg.build_specialists", return_value={}),
        patch("src.jobs.manager.react_pkg.build_specialists", return_value={}),
        patch("src.jobs.manager.get_storage_client") as mk_storage,
    ):
        from src.storage.local_storage import LocalStorageClient
        # Real local storage for save_/load_/delete_parked_state.
        real_storage = LocalStorageClient()
        # Wrap update_job + save_traces as AsyncMocks so we can assert on them
        # without needing the jobs.json shape to be fully populated.
        real_storage.update_job = AsyncMock(return_value=True)
        real_storage.save_traces = AsyncMock()
        mk_storage.return_value = real_storage

        from src.jobs.manager import JobManager

        manager = JobManager(orchestrator_mode="standard")
        yield manager, real_storage

    settings_mod.get_settings.cache_clear()


def _parked_state() -> AnalysisState:
    state = AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(
            url="kaggle.com/x", user_provided_context="initial analyst notes"
        ),
        treatment_variable="t",
        outcome_variable="y",
        status=JobStatus.AWAITING_APPROVAL,
        refined_dag=CausalDAG(
            nodes=["t", "y", "x1", "x2"],
            edges=[CausalEdge(source="t", target="y"), CausalEdge(source="x1", target="y")],
            discovery_method="domain_expert_fusion",
            adjustment_set=["x1"],  # original choice from dag_expert
        ),
        eda_result=EDAResult(data_quality_score=82.0),
    )
    return state


# --- APPROVED end-to-end ---------------------------------------------------


@pytest.mark.asyncio
async def test_approved_path_merges_edits_clears_park_and_respawns(
    manager_against_real_local_storage,
):
    manager, storage = manager_against_real_local_storage

    # 1. Park a state on disk
    await storage.save_parked_state(_parked_state())

    # 2. Build an approval with DAG edits + appended context
    approval = HumanApproval.approve(
        granted_by="reviewer@example.com",
        dag_edits=DagEdit(
            adjustment_set=["x1", "x2"],  # user added x2
            variable_roles={"x1": "confounder", "x2": "confounder"},
        ),
        appended_context="x2 is a known confounder in this study",
    )

    # 3. Capture the respawned task's state without actually running it
    captured: dict = {}

    def fake_create_task(coro):
        # Discard the coroutine to keep the test deterministic. We assert
        # that resume_from_approval *intended* to spawn it, and that the
        # state it set up matches the post-edit contract.
        coro.close()
        captured["task_was_created"] = True
        task = MagicMock()
        task.cancel = MagicMock()
        return task

    with patch("src.jobs.manager.asyncio.create_task", side_effect=fake_create_task):
        result = await manager.resume_from_approval("job-1", approval)

    # 4. Resumed, parked record deleted, status carried back to discovery
    assert result == {"resumed": True, "status": "discovering_causal"}
    assert captured.get("task_was_created") is True
    assert await storage.load_parked_state("job-1") is None
    storage.update_job.assert_awaited()

    # 5. Inspect the state the respawn was set up with by reloading what
    #    the manager passed into create_task. We tracked task_was_created
    #    only; to inspect the state we drive resume again with a different
    #    capture that records the state argument.
    assert "task_was_created" in captured


@pytest.mark.asyncio
async def test_approved_path_state_carries_edits_into_respawn(
    manager_against_real_local_storage,
):
    """The post-edit state passed into asyncio.create_task must reflect
    the edited adjustment_set and the appended user_provided_context —
    that is the contract effect_estimator depends on at the top of its
    next loop iteration."""
    manager, storage = manager_against_real_local_storage
    await storage.save_parked_state(_parked_state())

    approval = HumanApproval.approve(
        granted_by="reviewer",
        dag_edits=DagEdit(adjustment_set=["x1", "x2"]),
        appended_context="x2 is a known confounder",
    )

    captured_state = {}

    def fake_create_task(coro):
        # Coroutines for _run_job(state) hold the state arg as a frame local.
        # Extract it via the closure rather than letting the coroutine run.
        # Easier path: inspect coro.cr_frame.f_locals after close.
        try:
            frame_locals = dict(coro.cr_frame.f_locals)
            captured_state["state"] = frame_locals.get("state")
        finally:
            coro.close()
        return MagicMock()

    with patch("src.jobs.manager.asyncio.create_task", side_effect=fake_create_task):
        await manager.resume_from_approval("job-1", approval)

    state = captured_state["state"]
    assert state is not None
    assert state.proposed_dag.adjustment_set == ["x1", "x2"]
    assert state.human_approval is not None
    assert state.human_approval.decision == ApprovalDecision.APPROVED
    assert "x2 is a known confounder" in state.dataset_info.user_provided_context
    # Existing context survived (appended, not overwritten)
    assert "initial analyst notes" in state.dataset_info.user_provided_context
    # SSE event the UI watches to flip out of the gate UI.
    assert any(e["event_type"] == "approval_granted" for e in state.sse_events)
    granted_event = next(
        e for e in state.sse_events if e["event_type"] == "approval_granted"
    )
    assert granted_event["data"]["decision"] == "approved"
    assert "adjustment_set" in granted_event["data"]["edited_fields"]
    assert "appended_context" in granted_event["data"]["edited_fields"]


# --- REJECTED end-to-end ---------------------------------------------------


@pytest.mark.asyncio
async def test_rejected_path_fails_job_and_does_not_respawn(
    manager_against_real_local_storage,
):
    manager, storage = manager_against_real_local_storage
    await storage.save_parked_state(_parked_state())

    approval = HumanApproval.reject(reason="adjustment set looks wrong; missing x3")

    with patch("src.jobs.manager.asyncio.create_task") as mk_task:
        result = await manager.resume_from_approval("job-1", approval)

    assert result == {"resumed": False, "status": "failed"}
    mk_task.assert_not_called()
    storage.save_traces.assert_awaited_once()
    # Parked record is cleaned up regardless of decision.
    assert await storage.load_parked_state("job-1") is None


# --- Missing parked state --------------------------------------------------


@pytest.mark.asyncio
async def test_resume_from_approval_raises_when_no_parked_state(
    manager_against_real_local_storage,
):
    manager, _ = manager_against_real_local_storage
    approval = HumanApproval.approve()
    with pytest.raises(ValueError, match="not parked"):
        await manager.resume_from_approval("does-not-exist", approval)
