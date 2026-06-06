"""LocalStorageClient parked-state round-trip.

The approval gate persists the full AnalysisState mid-run, then reloads
it after the human approves. These tests pin that round-trip preserves
the fields downstream code reads — DAG, EDA, briefs, human_approval —
and that delete is idempotent.

Firestore round-trip lives in integration tests (needs emulator); the
local backend covers shape and idempotency.
"""
from __future__ import annotations

import pytest

from src.analysis.agents.base.state import AnalysisState, DatasetInfo, JobStatus
from src.analysis.agents.causal_discovery.output import CausalDAG, CausalEdge
from src.analysis.agents.eda.output import EDAResult
from src.domain.approval import DagEdit, HumanApproval
from src.domain.briefs import AgentBrief, Flag
from src.storage.local_storage import LocalStorageClient


@pytest.fixture
def storage(tmp_path, monkeypatch) -> LocalStorageClient:
    """LocalStorageClient pointed at a temp directory so each test is isolated."""
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    # Clear cached Settings so the new env var takes effect
    from src.config import settings as settings_mod
    settings_mod.get_settings.cache_clear()
    client = LocalStorageClient()
    yield client
    settings_mod.get_settings.cache_clear()


def _gate_state(job_id: str = "job-1") -> AnalysisState:
    state = AnalysisState(
        job_id=job_id,
        dataset_info=DatasetInfo(url="kaggle.com/x", user_provided_context="initial notes"),
        treatment_variable="t",
        outcome_variable="y",
        status=JobStatus.AWAITING_APPROVAL,
        refined_dag=CausalDAG(
            nodes=["t", "y", "x1"],
            edges=[CausalEdge(source="t", target="y"), CausalEdge(source="x1", target="y")],
            discovery_method="domain_expert_fusion",
            interpretation="x1 confounds t→y",
            adjustment_set=["x1"],
            variable_roles={"x1": "confounder"},
        ),
        eda_result=EDAResult(
            data_quality_score=82.0,
            data_quality_issues=["small control group"],
            balance_summary="moderate imbalance",
        ),
    )
    state.agent_briefs["eda_agent"] = AgentBrief(
        agent="eda_agent",
        status="done",
        headline="moderate treatment-control imbalance",
        flags=[Flag.TC_IMBALANCE],
        raised_issues=["control ~20% smaller than treated"],
    )
    return state


# --- save → load round-trip preserves the gate snapshot --------------------


@pytest.mark.asyncio
async def test_round_trip_preserves_dag_eda_and_briefs(storage):
    original = _gate_state()
    await storage.save_parked_state(original)
    restored = await storage.load_parked_state("job-1")

    assert restored is not None
    assert restored.job_id == "job-1"
    assert restored.status == JobStatus.AWAITING_APPROVAL
    assert restored.refined_dag.adjustment_set == ["x1"]
    assert restored.refined_dag.variable_roles == {"x1": "confounder"}
    assert restored.eda_result.data_quality_score == 82.0
    assert "eda_agent" in restored.agent_briefs
    assert restored.agent_briefs["eda_agent"].flags == [Flag.TC_IMBALANCE]


@pytest.mark.asyncio
async def test_round_trip_preserves_human_approval_when_set(storage):
    """Resume case: state already has an APPROVED decision attached when
    the worker writes it back (defence in depth)."""
    state = _gate_state()
    state.human_approval = HumanApproval.approve(
        granted_by="r",
        dag_edits=DagEdit(adjustment_set=["age", "education"]),
        appended_context="extra notes",
    )
    await storage.save_parked_state(state)
    restored = await storage.load_parked_state("job-1")
    assert restored.human_approval is not None
    assert restored.human_approval.granted_by == "r"
    assert restored.human_approval.dag_edits.adjustment_set == ["age", "education"]


# --- load missing returns None --------------------------------------------


@pytest.mark.asyncio
async def test_load_missing_returns_none(storage):
    assert await storage.load_parked_state("does-not-exist") is None


# --- delete is idempotent --------------------------------------------------


@pytest.mark.asyncio
async def test_delete_returns_true_when_present_false_when_missing(storage):
    state = _gate_state()
    await storage.save_parked_state(state)
    assert await storage.delete_parked_state("job-1") is True
    assert await storage.delete_parked_state("job-1") is False
    assert await storage.load_parked_state("job-1") is None


# --- save is keyed by job_id (multiple jobs coexist) -----------------------


@pytest.mark.asyncio
async def test_two_jobs_coexist_in_store(storage):
    a = _gate_state(job_id="job-a")
    b = _gate_state(job_id="job-b")
    await storage.save_parked_state(a)
    await storage.save_parked_state(b)
    restored_a = await storage.load_parked_state("job-a")
    restored_b = await storage.load_parked_state("job-b")
    assert restored_a.job_id == "job-a"
    assert restored_b.job_id == "job-b"


# --- save overwrites last writer wins -------------------------------------


@pytest.mark.asyncio
async def test_resave_overwrites_existing_entry(storage):
    state = _gate_state()
    await storage.save_parked_state(state)
    state.eda_result = EDAResult(data_quality_score=99.0)
    await storage.save_parked_state(state)
    restored = await storage.load_parked_state("job-1")
    assert restored.eda_result.data_quality_score == 99.0
