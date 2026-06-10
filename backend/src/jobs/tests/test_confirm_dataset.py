"""JobManager.confirm_dataset: accept the dataset at the data gate and stop.

Confirming marks the job CONFIRMED, keeps the parked state as the confirmed
record, and does NOT respawn the pipeline. Only valid at the data gate.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis.agents.base import (
    AnalysisState,
    DataProfile,
    DatasetInfo,
    JobStatus,
)


@pytest.fixture()
def manager():
    fake_settings = MagicMock()
    fake_settings.max_concurrent_jobs = 2
    fake_settings.agent_timeout_seconds = 300
    fake_settings.job_timeout_multiplier = 4
    fake_settings.instance_id = "test1234"

    with (
        patch("src.config.get_settings", return_value=fake_settings),
        patch("src.jobs.manager.get_settings", return_value=fake_settings),
        patch("src.jobs.manager.get_storage_client") as mock_storage,
        patch("src.jobs.manager.standard_pkg.build_specialists", return_value={}),
        patch("src.jobs.manager.react_pkg.build_specialists", return_value={}),
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
        treatment_variable="treat",
        outcome_variable="re78",
    )
    state.status = JobStatus.AWAITING_APPROVAL
    state.data_profile = DataProfile(
        n_samples=10,
        n_features=2,
        feature_names=["treat", "re78"],
        feature_types={"treat": "binary", "re78": "numeric"},
        missing_values={},
        numeric_stats={},
        categorical_stats={},
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
async def test_confirm_rejected_at_a_later_gate(manager):
    mgr, storage = manager
    state = _data_gate_state()
    storage.load_parked_state = AsyncMock(return_value=state)

    with patch(
        "src.analysis.orchestrator.base.should_pause_for_dag_approval",
        return_value=True,
    ):
        with pytest.raises(ValueError, match="data-review gate"):
            await mgr.confirm_dataset("job-confirm")
    storage.save_parked_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_without_parked_state_raises(manager):
    mgr, storage = manager
    storage.load_parked_state = AsyncMock(return_value=None)
    with pytest.raises(ValueError):
        await mgr.confirm_dataset("job-confirm")


@pytest.mark.asyncio
async def test_confirm_rejected_when_treatment_is_not_a_real_column(manager):
    mgr, storage = manager
    state = _data_gate_state()
    state.treatment_variable = "typo"  # not in feature_names
    storage.load_parked_state = AsyncMock(return_value=state)

    with pytest.raises(ValueError, match="not a column"):
        await mgr.confirm_dataset("job-confirm")

    # The dataset never reaches CONFIRMED on invalid inputs (format contract s9).
    assert state.status == JobStatus.AWAITING_APPROVAL
    storage.save_parked_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_rejected_when_a_required_input_is_missing(manager):
    mgr, storage = manager
    state = _data_gate_state()
    state.outcome_variable = None  # required, never set
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
async def test_run_analysis_respawns_and_clears_parked(manager):
    mgr, storage = manager
    state = _data_gate_state()
    state.status = JobStatus.CONFIRMED
    storage.load_parked_state = AsyncMock(return_value=state)

    with patch.object(mgr, "_run_job", new=AsyncMock()) as run_job:
        result = await mgr.run_analysis("job-confirm")
        run_job.assert_called_once()

    storage.delete_parked_state.assert_awaited_once()
    assert result["resumed"] is True
    # Clean up the scheduled task so it doesn't leak past the test.
    task = mgr._running_jobs.get("job-confirm")
    if task is not None:
        task.cancel()


@pytest.mark.asyncio
async def test_run_analysis_without_confirmed_state_raises(manager):
    mgr, storage = manager
    storage.load_parked_state = AsyncMock(return_value=None)
    with pytest.raises(ValueError):
        await mgr.run_analysis("job-confirm")
