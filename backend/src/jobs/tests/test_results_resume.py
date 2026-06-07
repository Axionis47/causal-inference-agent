"""Results gate resume routing: approve / revise / reject (checkpoint B).

Drives JobManager.resume_from_approval for a job parked at the results gate
(data + DAG approved, estimates + sensitivity present, results_approval unset).
Pins that the decision lands on results_approval, that REVISE clears the
estimates so the estimation tail re-runs (bounded), and that the respawned state
matches what effect_estimator reads next.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis.agents.base.state import AnalysisState, DatasetInfo, JobStatus
from src.analysis.agents.effect_estimator.output import TreatmentEffectResult
from src.analysis.agents.sensitivity_analyst.output import SensitivityResult
from src.domain.approval import ApprovalDecision, HumanApproval
from src.jobs.gate_resume import MAX_GATE_REVISIONS, RevisionLimitReached


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


def _results_parked_state(revision_count: int = 0) -> AnalysisState:
    return AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(url="kaggle.com/x", user_provided_context="initial notes"),
        treatment_variable="t",
        outcome_variable="y",
        status=JobStatus.AWAITING_APPROVAL,
        human_approval=HumanApproval.approve(),
        dag_approval=HumanApproval.approve(),
        treatment_effects=[TreatmentEffectResult(method="OLS", estimand="ATE", estimate=1.0,
                                                 std_error=0.1, ci_lower=0.8, ci_upper=1.2)],
        sensitivity_results=[SensitivityResult(method="evalue", robustness_value=1.8, interpretation="ok")],
        results_revision_count=revision_count,
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
async def test_results_approve_lands_on_results_approval_and_proceeds_to_critique(manager_against_real_local_storage):
    manager, storage = manager_against_real_local_storage
    await storage.save_parked_state(_results_parked_state())
    approval = HumanApproval.approve(appended_context="the IPW estimate looks right")
    captured: dict = {}

    with patch("src.jobs.gate_resume.asyncio.create_task", side_effect=_capture_respawn(captured)):
        result = await manager.resume_from_approval("job-1", approval)

    assert result == {"resumed": True, "status": "critique_review"}
    s = captured["state"]
    assert s.results_approval is not None and s.results_approval.decision == ApprovalDecision.APPROVED
    # The estimates are kept on approve (only revise clears them).
    assert len(s.treatment_effects) == 1
    assert any(e["event_type"] == "results_approval_granted" for e in s.sse_events)
    assert await storage.load_parked_state("job-1") is None


@pytest.mark.asyncio
async def test_results_revise_clears_the_estimates_for_re_estimation(manager_against_real_local_storage):
    manager, storage = manager_against_real_local_storage
    await storage.save_parked_state(_results_parked_state())
    approval = HumanApproval.revise(appended_context="re-estimate excluding the 2009 cohort")
    captured: dict = {}

    with patch("src.jobs.gate_resume.asyncio.create_task", side_effect=_capture_respawn(captured)):
        result = await manager.resume_from_approval("job-1", approval)

    assert result["resumed"] is True
    s = captured["state"]
    assert s.treatment_effects == []        # cleared so effect_estimator re-runs
    assert s.sensitivity_results == []       # cleared so sensitivity re-runs
    assert s.results_approval is None         # gate will fire again on fresh estimates
    assert s.results_revision_count == 1
    assert "re-estimate excluding the 2009 cohort" in s.dataset_info.user_provided_context
    assert any(e["event_type"] == "results_revision_requested" for e in s.sse_events)


@pytest.mark.asyncio
async def test_results_revise_at_the_cap_raises_and_stays_parked(manager_against_real_local_storage):
    manager, storage = manager_against_real_local_storage
    await storage.save_parked_state(_results_parked_state(revision_count=MAX_GATE_REVISIONS))
    approval = HumanApproval.revise(appended_context="one more change")

    with patch("src.jobs.gate_resume.asyncio.create_task") as mk_task:
        with pytest.raises(RevisionLimitReached):
            await manager.resume_from_approval("job-1", approval)

    mk_task.assert_not_called()
    assert await storage.load_parked_state("job-1") is not None


@pytest.mark.asyncio
async def test_results_reject_fails_the_job_without_respawn(manager_against_real_local_storage):
    manager, storage = manager_against_real_local_storage
    await storage.save_parked_state(_results_parked_state())
    approval = HumanApproval.reject(reason="effect size is implausibly large")

    with patch("src.jobs.gate_resume.asyncio.create_task") as mk_task:
        result = await manager.resume_from_approval("job-1", approval)

    assert result == {"resumed": False, "status": "failed"}
    mk_task.assert_not_called()
    storage.save_traces.assert_awaited()
    assert await storage.load_parked_state("job-1") is None
