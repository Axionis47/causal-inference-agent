"""JobManager.confirm_dataset, resume_from_approval, and run_analysis at the
data gate.

Confirming requires a causal question (treatment and outcome are no longer named
by the human), marks the job CONFIRMED, keeps the parked state as the confirmed
record, and does not start the pipeline. resume_from_approval routes an APPROVED
decision through confirm_dataset and fails the job on REJECTED, clearing the
parked state. run_analysis hands the confirmed record to the analysis runner,
which flips the job to running_analysis and spawns the spine task; the parked
state is kept as the confirmed record.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis_v2.state import (
    AnalysisState,
    DataProfile,
    DatasetInfo,
    JobStatus,
)


@pytest.fixture()
def manager():
    fake_settings = MagicMock()
    fake_settings.max_concurrent_jobs = 2
    fake_settings.instance_id = "test1234"

    with (
        patch("src.jobs.manager.get_settings", return_value=fake_settings),
        patch("src.jobs.manager.get_storage_client") as mock_storage,
    ):
        storage = MagicMock()
        storage.save_parked_state = AsyncMock()
        storage.update_job = AsyncMock(return_value=True)
        storage.delete_parked_state = AsyncMock()
        mock_storage.return_value = storage

        from src.jobs.manager import JobManager

        yield JobManager(orchestrator_mode="standard"), storage


def _data_gate_state() -> AnalysisState:
    state = AnalysisState(
        job_id="job-confirm",
        dataset_info=DatasetInfo(url="https://www.kaggle.com/datasets/o/n"),
        causal_question="Does training raise income?",
    )
    state.status = JobStatus.AWAITING_APPROVAL
    state.data_profile = DataProfile(
        n_samples=10,
        n_features=2,
        feature_names=["treat", "re78"],
        feature_types={"treat": "binary", "re78": "numeric"},
        missing_values={},
    )
    return state


@pytest.mark.asyncio
async def test_confirm_marks_confirmed_keeps_parked_no_respawn(manager):
    mgr, storage = manager
    state = _data_gate_state()
    storage.load_parked_state = AsyncMock(return_value=state)

    result = await mgr.confirm_dataset("job-confirm")

    assert state.status == JobStatus.CONFIRMED
    assert result["status"] == "confirmed"
    # The confirmed-dataset record is kept (read by the view + run_analysis),
    # not deleted, and the pipeline is not respawned.
    storage.save_parked_state.assert_awaited_once()
    storage.delete_parked_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_without_parked_state_raises(manager):
    mgr, storage = manager
    storage.load_parked_state = AsyncMock(return_value=None)
    with pytest.raises(ValueError):
        await mgr.confirm_dataset("job-confirm")


@pytest.mark.asyncio
async def test_confirm_rejected_when_question_is_missing(manager):
    mgr, storage = manager
    state = _data_gate_state()
    state.causal_question = None  # required, never set
    storage.load_parked_state = AsyncMock(return_value=state)

    with pytest.raises(ValueError, match="required"):
        await mgr.confirm_dataset("job-confirm")

    assert state.status == JobStatus.AWAITING_APPROVAL
    storage.save_parked_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_rejected_when_time_dimension_lacks_a_column(manager):
    mgr, storage = manager
    state = _data_gate_state()
    state.data_profile.has_time_dimension = True
    state.data_profile.time_column = None  # claimed but absent
    storage.load_parked_state = AsyncMock(return_value=state)

    with pytest.raises(ValueError, match="time_column"):
        await mgr.confirm_dataset("job-confirm")

    assert state.status == JobStatus.AWAITING_APPROVAL
    storage.save_parked_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_reject_at_gate_marks_failed_and_clears_parked_state(manager):
    from src.domain.approval import HumanApproval

    mgr, storage = manager
    state = _data_gate_state()
    storage.load_parked_state = AsyncMock(return_value=state)

    result = await mgr.resume_from_approval(
        "job-confirm", HumanApproval.reject(reason="wrong dataset")
    )

    assert result == {"resumed": False, "status": "failed"}
    assert state.status == JobStatus.FAILED
    assert "wrong dataset" in (state.error_message or "")
    storage.update_job.assert_awaited_once()
    storage.delete_parked_state.assert_awaited_once_with("job-confirm")


@pytest.mark.asyncio
async def test_approval_decision_delegates_to_confirm(manager):
    from src.domain.approval import HumanApproval

    mgr, storage = manager
    state = _data_gate_state()
    storage.load_parked_state = AsyncMock(return_value=state)

    result = await mgr.resume_from_approval("job-confirm", HumanApproval.approve())

    assert result == {"resumed": False, "status": "confirmed"}
    assert state.status == JobStatus.CONFIRMED
    # Confirm keeps the record; nothing is deleted and nothing respawns.
    storage.save_parked_state.assert_awaited_once()
    storage.delete_parked_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_analysis_hands_the_confirmed_record_to_the_runner(manager):
    mgr, storage = manager
    state = _data_gate_state()
    state.status = JobStatus.CONFIRMED
    storage.load_parked_state = AsyncMock(return_value=state)

    launched = AsyncMock(return_value={"resumed": True, "status": "running_analysis"})
    with patch("src.analysis_v2.runner.start", new=launched):
        result = await mgr.run_analysis("job-confirm")

    launched.assert_awaited_once_with(state, mgr)
    storage.delete_parked_state.assert_not_awaited()
    assert result == {"resumed": True, "status": "running_analysis"}


@pytest.mark.asyncio
async def test_run_analysis_is_idempotent_while_the_job_task_is_live(manager):
    mgr, storage = manager
    state = _data_gate_state()
    state.status = JobStatus.CONFIRMED
    storage.load_parked_state = AsyncMock(return_value=state)
    mgr._running_jobs["job-confirm"] = MagicMock()

    launched = AsyncMock()
    with patch("src.analysis_v2.runner.start", new=launched):
        result = await mgr.run_analysis("job-confirm")

    launched.assert_not_awaited()
    assert result == {"resumed": False, "status": "confirmed"}


@pytest.mark.asyncio
async def test_run_analysis_without_confirmed_state_raises(manager):
    mgr, storage = manager
    storage.load_parked_state = AsyncMock(return_value=None)
    with pytest.raises(ValueError):
        await mgr.run_analysis("job-confirm")
