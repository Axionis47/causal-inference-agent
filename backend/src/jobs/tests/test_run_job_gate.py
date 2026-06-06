"""JobManager data-review gate: download-and-park before any analysis.

On a job's first run (no human approval yet) the worker downloads the
dataset, parks at AWAITING_APPROVAL, and returns WITHOUT running the
orchestrator. This is what keeps every label-producing step out of the
pre-approval path: the orchestrator (where all inference lives) does not
run until the human approves. A download failure fails the job and never
parks.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.analysis.agents.base.state import (
    AnalysisState,
    DatasetInfo,
    FileEntry,
    JobStatus,
)


@pytest.fixture()
def manager_with_mocks():
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
        storage.update_job = AsyncMock(return_value=True)
        storage.save_results = AsyncMock()
        storage.save_traces = AsyncMock()
        storage.save_parked_state = AsyncMock()
        mock_storage.return_value = storage

        from src.jobs.manager import JobManager

        manager = JobManager(orchestrator_mode="standard")
        yield manager, storage


def _fresh_state() -> AnalysisState:
    """A job as first submitted: no human approval yet."""
    return AnalysisState(
        job_id="job-gate",
        dataset_info=DatasetInfo(url="https://www.kaggle.com/datasets/o/n"),
    )


@pytest.mark.asyncio
async def test_first_run_downloads_parks_and_does_not_run_orchestrator(manager_with_mocks):
    manager, storage = manager_with_mocks

    async def fake_load(state: AnalysisState):
        # Simulate the download: files land on state, manifest written on disk.
        state.dataset_info.files = [
            FileEntry(name="d.csv", size_bytes=10, format="csv", used=True)
        ]
        return pd.DataFrame({"a": [1, 2]}), None

    with (
        patch(
            "src.analysis.agents.data_profiler.loading.load_dataset",
            side_effect=fake_load,
        ),
        patch.object(manager, "_create_orchestrator") as mk_create,
    ):
        await manager._run_job_inner(_fresh_state())

        # The orchestrator must not be built or run before approval.
        mk_create.assert_not_called()

    # The job parked: full state saved, no results.
    storage.save_parked_state.assert_awaited_once()
    storage.save_results.assert_not_awaited()
    parked_state = storage.save_parked_state.await_args.args[0]
    assert parked_state.status == JobStatus.AWAITING_APPROVAL


@pytest.mark.asyncio
async def test_download_failure_fails_job_and_does_not_park(manager_with_mocks):
    manager, storage = manager_with_mocks

    async def fake_load(state: AnalysisState):
        return None, "Dataset 'o/n' is not accessible."

    with (
        patch(
            "src.analysis.agents.data_profiler.loading.load_dataset",
            side_effect=fake_load,
        ),
        patch.object(manager, "_create_orchestrator") as mk_create,
    ):
        await manager._run_job_inner(_fresh_state())
        mk_create.assert_not_called()

    storage.save_parked_state.assert_not_awaited()
    # Job row updated to FAILED with the download error surfaced.
    final = storage.update_job.await_args.args[0]
    assert final.status == JobStatus.FAILED
    assert "not accessible" in (final.error_message or "")
