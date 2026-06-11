"""JobManager.set_dataset_inputs: analyst edits at the data gate.

The analyst can refine the causal question and set or clear the time column while
the job is parked. Treatment and outcome are no longer chosen here. The time
column is validated against the profiled dataset's real columns, and edits are
only allowed before the data gate is approved.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis_v2.state import AnalysisState, DataProfile, DatasetInfo
from src.domain.approval import ApprovalDecision, HumanApproval


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
        mock_storage.return_value = storage

        from src.jobs.manager import JobManager

        yield JobManager(orchestrator_mode="standard"), storage


def _parked_state(approved: bool = False) -> AnalysisState:
    state = AnalysisState(
        job_id="job-inputs",
        dataset_info=DatasetInfo(url="https://www.kaggle.com/datasets/o/n"),
        causal_question="Does training raise income?",
    )
    state.data_profile = DataProfile(
        n_samples=10,
        n_features=4,
        feature_names=["treat", "re78", "age", "date"],
        feature_types={
            "treat": "binary",
            "re78": "numeric",
            "age": "numeric",
            "date": "datetime",
        },
        missing_values={},
    )
    if approved:
        state.human_approval = HumanApproval(
            decision=ApprovalDecision.APPROVED,
            granted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    return state


@pytest.mark.asyncio
async def test_refines_question_and_sets_time(manager):
    mgr, storage = manager
    state = _parked_state()
    with patch.object(mgr, "get_parked_state", AsyncMock(return_value=state)):
        result = await mgr.set_dataset_inputs(
            "job-inputs",
            causal_question="Does training raise earnings?",
            time_column="date",
        )
    assert state.causal_question == "Does training raise earnings?"
    assert state.data_profile.time_column == "date"
    assert state.data_profile.has_time_dimension is True
    storage.save_parked_state.assert_awaited_once()
    assert result == {
        "causal_question": "Does training raise earnings?",
        "time_column": "date",
        "has_time_dimension": True,
    }


@pytest.mark.asyncio
async def test_clearing_the_time_column_leaves_the_question(manager):
    mgr, _ = manager
    state = _parked_state()
    state.data_profile.has_time_dimension = True
    state.data_profile.time_column = "date"
    with patch.object(mgr, "get_parked_state", AsyncMock(return_value=state)):
        result = await mgr.set_dataset_inputs(
            "job-inputs",
            causal_question=None,
            time_column=None,
        )
    assert state.data_profile.has_time_dimension is False
    assert state.data_profile.time_column is None
    assert result["has_time_dimension"] is False
    # A None question is a no-op, the existing question is left in place.
    assert state.causal_question == "Does training raise income?"


@pytest.mark.asyncio
async def test_unknown_time_column_is_rejected_and_nothing_persists(manager):
    mgr, storage = manager
    state = _parked_state()
    with patch.object(mgr, "get_parked_state", AsyncMock(return_value=state)):
        with pytest.raises(ValueError, match="not a column"):
            await mgr.set_dataset_inputs(
                "job-inputs",
                causal_question=None,
                time_column="nope",
            )
    storage.save_parked_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_blank_question_is_rejected_and_nothing_persists(manager):
    mgr, storage = manager
    state = _parked_state()
    with patch.object(mgr, "get_parked_state", AsyncMock(return_value=state)):
        with pytest.raises(ValueError, match="blank"):
            await mgr.set_dataset_inputs(
                "job-inputs",
                causal_question="   ",
                time_column=None,
            )
    storage.save_parked_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejected_past_the_data_gate(manager):
    mgr, _ = manager
    state = _parked_state(approved=True)
    with patch.object(mgr, "get_parked_state", AsyncMock(return_value=state)):
        with pytest.raises(ValueError, match="data-review gate"):
            await mgr.set_dataset_inputs(
                "job-inputs",
                causal_question="Anything?",
                time_column=None,
            )
