"""JobManager._run_job_inner: post-approval (orchestrator) branches.

These cover what the worker does AFTER the human has approved at the
data-review gate, when the orchestrator actually runs. The states here
carry an APPROVED human_approval so the worker skips the download gate
(covered in test_run_job_gate.py) and enters the orchestrator. The
fallback AWAITING_APPROVAL branch (orchestrator yields the gate itself)
must still persist the full state, save partial traces, and skip
save_results.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis.agents.base.state import (
    AgentTrace,
    AnalysisState,
    DatasetInfo,
    JobStatus,
)
from src.domain.approval import HumanApproval


@pytest.fixture()
def manager_with_mocks():
    """Build a JobManager whose orchestrator+storage are mocked so we
    can drive _run_job_inner directly without spinning up real agents."""
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
        storage.create_job = AsyncMock()
        storage.create_job_if_capacity = AsyncMock(return_value=True)
        storage.update_job = AsyncMock(return_value=True)
        storage.save_results = AsyncMock()
        storage.save_traces = AsyncMock()
        storage.save_parked_state = AsyncMock()
        mock_storage.return_value = storage

        from src.jobs.manager import JobManager

        manager = JobManager(orchestrator_mode="standard")
        yield manager, storage


def _state_with_traces() -> AnalysisState:
    """Approved state the worker resumes after the gate (orchestrator runs)."""
    state = AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(url="kaggle.com/x"),
        treatment_variable="t",
        outcome_variable="y",
    )
    state.human_approval = HumanApproval.approve()
    return state


def _parked_final_state() -> AnalysisState:
    """Final state the orchestrator returns when the gate fires."""
    state = _state_with_traces()
    state.status = JobStatus.AWAITING_APPROVAL
    # Trace from dag_expert: realistic, because the gate fires right after.
    state.add_trace(AgentTrace(agent_name="dag_expert", action="completed"))
    return state


# --- The parked branch -----------------------------------------------------


@pytest.mark.asyncio
async def test_parked_branch_persists_full_state_not_results(manager_with_mocks):
    manager, storage = manager_with_mocks

    parked = _parked_final_state()

    with patch.object(manager, "_create_orchestrator") as mk_create:
        orch = MagicMock()
        orch.set_status_callback = MagicMock()
        orch.execute_with_tracing = AsyncMock(return_value=parked)
        mk_create.return_value = orch

        await manager._run_job_inner(_state_with_traces())

    # Parked state was persisted; results path was NOT.
    storage.save_parked_state.assert_awaited_once()
    storage.save_results.assert_not_awaited()
    # Partial traces are still useful — they should be saved.
    storage.save_traces.assert_awaited_once()
    # Job row updated so the API can find the parked job.
    assert storage.update_job.await_count >= 1


@pytest.mark.asyncio
async def test_completed_branch_still_saves_results_not_parked_state(manager_with_mocks):
    """Regression guard: completed jobs must keep the existing save path
    and must NOT touch the parked-state store."""
    manager, storage = manager_with_mocks

    final = _state_with_traces()
    final.status = JobStatus.COMPLETED

    with patch.object(manager, "_create_orchestrator") as mk_create:
        orch = MagicMock()
        orch.set_status_callback = MagicMock()
        orch.execute_with_tracing = AsyncMock(return_value=final)
        mk_create.return_value = orch

        await manager._run_job_inner(_state_with_traces())

    storage.save_results.assert_awaited_once()
    storage.save_parked_state.assert_not_awaited()


# --- Progress map extension ------------------------------------------------


def test_calculate_progress_maps_awaiting_approval(manager_with_mocks):
    manager, _ = manager_with_mocks
    # Sits between discovering_causal (44) and estimating_effects (56).
    pct = manager._calculate_progress("awaiting_approval")
    assert 44 < pct < 56
