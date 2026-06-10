"""JobManager.set_dataset_inputs: analyst corrections at the data gate.

The analyst can fix treatment / outcome / time in the preview while the job is
parked. The chosen names are validated against the profiled dataset's real
columns, and only before the data gate is approved.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis.agents.base import AnalysisState, DataProfile, DatasetInfo
from src.domain.approval import ApprovalDecision, HumanApproval


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
        mock_storage.return_value = storage

        from src.jobs.manager import JobManager

        yield JobManager(orchestrator_mode="standard"), storage


def _parked_state(approved: bool = False) -> AnalysisState:
    state = AnalysisState(
        job_id="job-inputs",
        dataset_info=DatasetInfo(url="https://www.kaggle.com/datasets/o/n"),
        treatment_variable="treat",
        outcome_variable="re78",
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
        numeric_stats={},
        categorical_stats={},
    )
    if approved:
        state.human_approval = HumanApproval(
            decision=ApprovalDecision.APPROVED,
            granted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    return state


@pytest.mark.asyncio
async def test_valid_correction_updates_and_persists(manager):
    mgr, storage = manager
    state = _parked_state()
    with patch.object(mgr, "get_parked_state", AsyncMock(return_value=state)):
        result = await mgr.set_dataset_inputs(
            "job-inputs",
            treatment_variable="age",
            outcome_variable="re78",
            time_column="date",
        )
    assert state.treatment_variable == "age"
    assert state.outcome_variable == "re78"
    assert state.data_profile.time_column == "date"
    assert state.data_profile.has_time_dimension is True
    storage.save_parked_state.assert_awaited_once()
    assert result == {
        "treatment_variable": "age",
        "outcome_variable": "re78",
        "time_column": "date",
        "has_time_dimension": True,
    }


@pytest.mark.asyncio
async def test_clearing_the_time_column(manager):
    mgr, _ = manager
    state = _parked_state()
    state.data_profile.has_time_dimension = True
    state.data_profile.time_column = "date"
    with patch.object(mgr, "get_parked_state", AsyncMock(return_value=state)):
        result = await mgr.set_dataset_inputs(
            "job-inputs",
            treatment_variable="treat",
            outcome_variable="re78",
            time_column=None,
        )
    assert state.data_profile.has_time_dimension is False
    assert state.data_profile.time_column is None
    assert result["has_time_dimension"] is False


@pytest.mark.asyncio
async def test_unknown_column_is_rejected_and_nothing_persists(manager):
    mgr, storage = manager
    state = _parked_state()
    with patch.object(mgr, "get_parked_state", AsyncMock(return_value=state)):
        with pytest.raises(ValueError, match="not a column"):
            await mgr.set_dataset_inputs(
                "job-inputs",
                treatment_variable="nope",
                outcome_variable="re78",
                time_column=None,
            )
    storage.save_parked_state.assert_not_awaited()
    # The state is left untouched on a rejected edit.
    assert state.treatment_variable == "treat"


@pytest.mark.asyncio
async def test_rejected_past_the_data_gate(manager):
    mgr, _ = manager
    state = _parked_state(approved=True)
    with patch.object(mgr, "get_parked_state", AsyncMock(return_value=state)):
        with pytest.raises(ValueError, match="data-review gate"):
            await mgr.set_dataset_inputs(
                "job-inputs",
                treatment_variable="age",
                outcome_variable="re78",
                time_column=None,
            )
