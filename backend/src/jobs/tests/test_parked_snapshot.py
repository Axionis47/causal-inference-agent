"""JobManager.get_parked_snapshot serves the data-gate payload.

This slice has exactly one gate (data review), so the snapshot is either None
(job not parked) or {"kind": "data", "payload": ...}. The payload carries the
relational profile when the bundle has one, so a client that lost the SSE
event to a refresh or a drop can rehydrate the review surface. The storage
layer is mocked so each test feeds one parked state and asserts the shape.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis_v2.state import AnalysisState, DatasetInfo, JobStatus
from src.domain.relational import FileRelational, RelationalProfile


@pytest.fixture
def manager_with_state():
    """Build a JobManager whose load_parked_state returns the given state."""

    def _make(state: AnalysisState | None):
        fake_settings = MagicMock()
        fake_settings.max_concurrent_jobs = 2
        fake_settings.instance_id = "test1234"
        with (
            patch("src.jobs.manager.get_settings", return_value=fake_settings),
            patch("src.jobs.manager.get_storage_client") as mk_storage,
        ):
            storage = MagicMock()
            storage.load_parked_state = AsyncMock(return_value=state)
            mk_storage.return_value = storage
            from src.jobs.manager import JobManager

            return JobManager(orchestrator_mode="standard")

    return _make


def _parked_state() -> AnalysisState:
    return AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(url="https://www.kaggle.com/datasets/o/n"),
        causal_question="Does training raise income?",
        status=JobStatus.AWAITING_APPROVAL,
    )


@pytest.mark.asyncio
async def test_snapshot_is_none_when_not_parked(manager_with_state):
    manager = manager_with_state(None)
    assert await manager.get_parked_snapshot("job-1") is None


@pytest.mark.asyncio
async def test_snapshot_names_the_data_gate_with_no_relational_profile(
    manager_with_state,
):
    manager = manager_with_state(_parked_state())
    snap = await manager.get_parked_snapshot("job-1")
    assert snap["kind"] == "data"
    assert snap["payload"] == {"relational": None}


@pytest.mark.asyncio
async def test_snapshot_carries_the_relational_profile_when_present(
    manager_with_state,
):
    state = _parked_state()
    state.relational_profile = RelationalProfile(
        shape_hint="same_schema_shards",
        files=[
            FileRelational(file="2020.csv", n_rows=100, n_cols=4),
            FileRelational(file="2021.csv", n_rows=120, n_cols=4),
        ],
        same_schema_groups=[["2020.csv", "2021.csv"]],
    )
    manager = manager_with_state(state)
    snap = await manager.get_parked_snapshot("job-1")
    assert snap["kind"] == "data"
    relational = snap["payload"]["relational"]
    assert relational["shape_hint"] == "same_schema_shards"
    assert [f["file"] for f in relational["files"]] == ["2020.csv", "2021.csv"]
