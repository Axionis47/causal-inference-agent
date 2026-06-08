"""JobManager.get_parked_snapshot picks the gate the parked state is at.

The snapshot used to always build the data-gate payload, so a job parked at
the DAG or results gate rehydrated as the data gate on refresh. These tests
pin that the snapshot now names the correct gate (data / dag / results) and
carries that gate's payload, built from the same functions that emit the SSE
events. The storage layer is mocked so each test feeds one realistic parked
state and asserts the chosen gate.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analysis.agents.base.state import AnalysisState, DatasetInfo, JobStatus
from src.analysis.agents.causal_discovery.output import CausalDAG, CausalEdge
from src.analysis.agents.effect_estimator.output import TreatmentEffectResult
from src.analysis.agents.sensitivity_analyst.output import SensitivityResult
from src.domain.approval import HumanApproval


@pytest.fixture
def manager_with_state():
    """Build a JobManager whose load_parked_state returns the given state."""

    def _make(state: AnalysisState | None):
        with (
            patch("src.jobs.manager.standard_pkg.build_specialists", return_value={}),
            patch("src.jobs.manager.react_pkg.build_specialists", return_value={}),
            patch("src.jobs.manager.get_storage_client") as mk_storage,
        ):
            storage = MagicMock()
            storage.load_parked_state = AsyncMock(return_value=state)
            mk_storage.return_value = storage
            from src.jobs.manager import JobManager

            return JobManager(orchestrator_mode="standard")

    return _make


def _base_state() -> AnalysisState:
    return AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(url="kaggle.com/x"),
        treatment_variable="t",
        outcome_variable="y",
        status=JobStatus.AWAITING_APPROVAL,
    )


def _dag() -> CausalDAG:
    return CausalDAG(
        nodes=["t", "y", "x1"],
        edges=[CausalEdge(source="t", target="y"), CausalEdge(source="x1", target="y")],
        discovery_method="dag_expert",
        adjustment_set=["x1"],
    )


@pytest.mark.asyncio
async def test_snapshot_is_none_when_not_parked(manager_with_state):
    manager = manager_with_state(None)
    assert await manager.get_parked_snapshot("job-1") is None


@pytest.mark.asyncio
async def test_snapshot_at_data_gate(manager_with_state):
    # No human_approval yet: the job is at the data gate.
    manager = manager_with_state(_base_state())
    snap = await manager.get_parked_snapshot("job-1")
    assert snap["kind"] == "data"
    assert snap["payload"]["treatment_variable"] == "t"


@pytest.mark.asyncio
async def test_snapshot_at_dag_gate(manager_with_state):
    # Data approved, a refined DAG exists, no estimates yet: the DAG gate.
    state = _base_state()
    state.human_approval = HumanApproval.approve()
    state.refined_dag = _dag()
    manager = manager_with_state(state)
    snap = await manager.get_parked_snapshot("job-1")
    assert snap["kind"] == "dag"
    assert snap["payload"]["dag"]["adjustment_set"] == ["x1"]


@pytest.mark.asyncio
async def test_snapshot_at_results_gate(manager_with_state):
    # DAG approved, estimates and sensitivity exist: the results gate. This is
    # the case the old data-only snapshot got wrong.
    state = _base_state()
    state.human_approval = HumanApproval.approve()
    state.dag_approval = HumanApproval.approve()
    state.refined_dag = _dag()
    state.treatment_effects = [
        TreatmentEffectResult(
            method="ipw",
            estimand="ATE",
            estimate=1.04,
            std_error=0.4,
            ci_lower=0.22,
            ci_upper=1.86,
            p_value=0.03,
        )
    ]
    state.sensitivity_results = [
        SensitivityResult(method="evalue", robustness_value=1.8, interpretation="moderate")
    ]
    manager = manager_with_state(state)
    snap = await manager.get_parked_snapshot("job-1")
    assert snap["kind"] == "results"
    assert snap["payload"]["effects"][0]["method"] == "ipw"
