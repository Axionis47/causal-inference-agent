"""DAG gate resume routing: approve / revise / reject (checkpoint A).

Drives JobManager.resume_from_approval against real local storage for a job
parked at the DAG gate (human_approval APPROVED, refined_dag present,
dag_approval unset). Pins that the decision lands on dag_approval (not
human_approval), that REVISE clears the DAG and bounds the redo loop, and that
the respawned state matches what dag_expert / effect_estimator will read next.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis.agents.base.state import AnalysisState, DatasetInfo, JobStatus
from src.analysis.agents.causal_discovery.output import CausalDAG, CausalEdge
from src.analysis.agents.eda.output import EDAResult
from src.domain.approval import ApprovalDecision, DagEdit, HumanApproval
from src.jobs.dag_resume import MAX_DAG_REVISIONS, RevisionLimitReached


@pytest.fixture()
def manager_against_real_local_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))

    from src.config import settings as settings_mod
    settings_mod.get_settings.cache_clear()
    settings_mod.get_settings()

    with (
        patch("src.jobs.manager.standard_pkg.build_specialists", return_value={}),
        patch("src.jobs.manager.react_pkg.build_specialists", return_value={}),
        patch("src.jobs.manager.get_storage_client") as mk_storage,
    ):
        from src.storage.local_storage import LocalStorageClient
        real_storage = LocalStorageClient()
        real_storage.update_job = AsyncMock(return_value=True)
        real_storage.save_traces = AsyncMock()
        mk_storage.return_value = real_storage

        from src.jobs.manager import JobManager

        manager = JobManager(orchestrator_mode="standard")
        yield manager, real_storage

    settings_mod.get_settings.cache_clear()


def _dag_parked_state(revision_count: int = 0) -> AnalysisState:
    return AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(url="kaggle.com/x", user_provided_context="initial notes"),
        treatment_variable="t",
        outcome_variable="y",
        status=JobStatus.AWAITING_APPROVAL,
        human_approval=HumanApproval.approve(),  # data gate already passed
        refined_dag=CausalDAG(
            nodes=["t", "y", "x1"],
            edges=[CausalEdge(source="t", target="y"), CausalEdge(source="x1", target="y")],
            discovery_method="dag_expert",
            adjustment_set=["x1"],
        ),
        eda_result=EDAResult(data_quality_score=80.0),
        dag_revision_count=revision_count,
    )


def _capture_respawn(captured: dict):
    def fake_create_task(coro):
        try:
            captured["state"] = dict(coro.cr_frame.f_locals).get("state")
        finally:
            coro.close()
        return MagicMock()
    return fake_create_task


@pytest.mark.asyncio
async def test_dag_approve_lands_on_dag_approval_and_respawns(manager_against_real_local_storage):
    manager, storage = manager_against_real_local_storage
    await storage.save_parked_state(_dag_parked_state())
    approval = HumanApproval.approve(
        dag_edits=DagEdit(adjustment_set=["x1", "x2"]),
        appended_context="x2 is a known confounder",
    )
    captured: dict = {}

    with patch("src.jobs.gate_resume.asyncio.create_task", side_effect=_capture_respawn(captured)):
        result = await manager.resume_from_approval("job-1", approval)

    assert result == {"resumed": True, "status": "discovering_causal"}
    s = captured["state"]
    # The DAG decision lands on dag_approval, NOT the data gate's human_approval.
    assert s.dag_approval is not None and s.dag_approval.decision == ApprovalDecision.APPROVED
    assert s.refined_dag.adjustment_set == ["x1", "x2"]  # structured edit applied
    assert "x2 is a known confounder" in s.dataset_info.user_provided_context
    assert "initial notes" in s.dataset_info.user_provided_context  # existing kept
    assert any(e["event_type"] == "dag_approval_granted" for e in s.sse_events)
    assert await storage.load_parked_state("job-1") is None


@pytest.mark.asyncio
async def test_dag_revise_clears_the_dag_and_keeps_the_gate_unapproved(manager_against_real_local_storage):
    manager, storage = manager_against_real_local_storage
    await storage.save_parked_state(_dag_parked_state())
    approval = HumanApproval.revise(appended_context="control for x3, it precedes treatment")
    captured: dict = {}

    with patch("src.jobs.gate_resume.asyncio.create_task", side_effect=_capture_respawn(captured)):
        result = await manager.resume_from_approval("job-1", approval)

    assert result["resumed"] is True
    s = captured["state"]
    assert s.refined_dag is None        # cleared so dag_expert re-runs with the note
    assert s.dag_approval is None        # gate will fire again on the redone DAG
    assert s.dag_revision_count == 1
    assert "control for x3" in s.dataset_info.user_provided_context
    assert any(e["event_type"] == "dag_revision_requested" for e in s.sse_events)
    assert await storage.load_parked_state("job-1") is None


@pytest.mark.asyncio
async def test_dag_revise_at_the_cap_raises_and_stays_parked(manager_against_real_local_storage):
    manager, storage = manager_against_real_local_storage
    await storage.save_parked_state(_dag_parked_state(revision_count=MAX_DAG_REVISIONS))
    approval = HumanApproval.revise(appended_context="one more change")

    with patch("src.jobs.gate_resume.asyncio.create_task") as mk_task:
        with pytest.raises(RevisionLimitReached):
            await manager.resume_from_approval("job-1", approval)

    mk_task.assert_not_called()
    # The job is left parked so the human can still approve or reject.
    assert await storage.load_parked_state("job-1") is not None


@pytest.mark.asyncio
async def test_dag_reject_fails_the_job_without_respawn(manager_against_real_local_storage):
    manager, storage = manager_against_real_local_storage
    await storage.save_parked_state(_dag_parked_state())
    approval = HumanApproval.reject(reason="the t -> x1 edge is backwards")

    with patch("src.jobs.gate_resume.asyncio.create_task") as mk_task:
        result = await manager.resume_from_approval("job-1", approval)

    assert result == {"resumed": False, "status": "failed"}
    mk_task.assert_not_called()
    storage.save_traces.assert_awaited()
    assert await storage.load_parked_state("job-1") is None


# --- REVISE decision contract (domain validator) ---------------------------


def test_revise_requires_appended_context():
    with pytest.raises(ValueError, match="appended_context is required"):
        HumanApproval(
            decision=ApprovalDecision.REVISE,
            granted_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )


def test_revise_constructor_builds_a_valid_record():
    rec = HumanApproval.revise(appended_context="add x3 as a confounder")
    assert rec.decision == ApprovalDecision.REVISE
    assert rec.appended_context == "add x3 as a confounder"
