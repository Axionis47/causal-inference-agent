"""Shared orchestrator gate: should_pause_for_approval + park_for_approval.

Pins the truth table both orchestrators must honour and the snapshot the
SSE event carries to the UI. Two orchestrators consuming a shared helper
is exactly the contract these tests guard.
"""
from __future__ import annotations

import pytest

from src.analysis.agents.base.state import AnalysisState, DatasetInfo, JobStatus
from src.analysis.agents.causal_discovery.output import CausalDAG, CausalEdge
from src.analysis.agents.eda.output import EDAResult
from src.analysis.agents.effect_estimator.output import TreatmentEffectResult
from src.analysis.orchestrator.base import (
    _build_gate_payload,
    park_for_approval,
    should_pause_for_approval,
)
from src.domain.approval import HumanApproval
from src.domain.briefs import AgentBrief, Flag


def _dag(
    *,
    nodes: list[str] | None = None,
    edges: list[CausalEdge] | None = None,
    adjustment_set: list[str] | None = None,
    variable_roles: dict[str, str] | None = None,
    forbidden_edges: list[dict[str, str]] | None = None,
) -> CausalDAG:
    return CausalDAG(
        nodes=nodes or ["t", "y", "x1"],
        edges=edges or [CausalEdge(source="t", target="y"), CausalEdge(source="x1", target="y")],
        discovery_method="domain_expert_fusion",
        interpretation="x1 confounds t→y",
        adjustment_set=adjustment_set,
        variable_roles=variable_roles,
        forbidden_edges=forbidden_edges,
    )


def _state(**overrides) -> AnalysisState:
    state = AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(url="kaggle.com/x"),
        treatment_variable="t",
        outcome_variable="y",
    )
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


# --- should_pause_for_approval truth table ---------------------------------


def test_no_dag_no_eda_does_not_pause():
    assert should_pause_for_approval(_state()) is False


def test_dag_only_does_not_pause_until_eda_lands():
    state = _state(refined_dag=_dag())
    assert should_pause_for_approval(state) is False


def test_eda_only_does_not_pause_until_dag_lands():
    state = _state(eda_result=EDAResult(data_quality_score=80.0))
    assert should_pause_for_approval(state) is False


def test_dag_plus_eda_pauses_when_no_approval():
    state = _state(refined_dag=_dag(), eda_result=EDAResult(data_quality_score=80.0))
    assert should_pause_for_approval(state) is True


def test_discovered_dag_alone_still_triggers_pause_when_eda_present():
    # refined_dag may be None if dag_expert was skipped; the gate should
    # still fire as long as *some* DAG and the EDA snapshot exist.
    state = _state(discovered_dag=_dag(), eda_result=EDAResult())
    assert should_pause_for_approval(state) is True


def test_prior_approved_decision_skips_pause():
    state = _state(
        refined_dag=_dag(),
        eda_result=EDAResult(),
        human_approval=HumanApproval.approve(),
    )
    assert should_pause_for_approval(state) is False


def test_prior_rejected_decision_does_not_skip_pause():
    # Defensive: if someone manually set REJECTED but the worker forgot
    # to fail the job, the gate should still hold (not silently proceed).
    state = _state(
        refined_dag=_dag(),
        eda_result=EDAResult(),
        human_approval=HumanApproval.reject(reason="DAG wrong"),
    )
    assert should_pause_for_approval(state) is True


def test_resume_with_estimation_in_progress_does_not_pause():
    # Once treatment_effects exist we are past the gate.
    state = _state(
        refined_dag=_dag(),
        eda_result=EDAResult(),
        treatment_effects=[
            TreatmentEffectResult(
                method="ols",
                estimand="ATE",
                estimate=1.0,
                std_error=0.1,
                ci_lower=0.8,
                ci_upper=1.2,
                p_value=0.01,
            )
        ],
    )
    assert should_pause_for_approval(state) is False


# --- _build_gate_payload shape --------------------------------------------


def test_gate_payload_carries_treatment_and_outcome():
    state = _state(refined_dag=_dag(), eda_result=EDAResult())
    payload = _build_gate_payload(state)
    assert payload["treatment_variable"] == "t"
    assert payload["outcome_variable"] == "y"


def test_gate_payload_eda_summary_pulls_quality_score_and_issues():
    state = _state(
        refined_dag=_dag(),
        eda_result=EDAResult(
            data_quality_score=72.5,
            data_quality_issues=[f"issue-{i}" for i in range(8)],
            balance_summary="treatment-control balance is moderate",
        ),
    )
    payload = _build_gate_payload(state)
    eda = payload["eda_summary"]
    assert eda["data_quality_score"] == 72.5
    assert eda["balance_summary"] == "treatment-control balance is moderate"
    assert len(eda["data_quality_issues"]) == 5  # capped at 5


def test_gate_payload_dag_prefers_refined_over_discovered():
    refined = _dag(adjustment_set=["x1"], variable_roles={"x1": "confounder"})
    discovered = _dag(adjustment_set=["x2"])
    state = _state(refined_dag=refined, discovered_dag=discovered, eda_result=EDAResult())
    payload = _build_gate_payload(state)
    assert payload["proposed_dag"]["adjustment_set"] == ["x1"]
    assert payload["proposed_dag"]["variable_roles"] == {"x1": "confounder"}


def test_gate_payload_dag_falls_back_to_discovered():
    state = _state(discovered_dag=_dag(adjustment_set=["x2"]), eda_result=EDAResult())
    payload = _build_gate_payload(state)
    assert payload["proposed_dag"]["adjustment_set"] == ["x2"]


def test_gate_payload_includes_brief_flags_for_sealed_agents():
    state = _state(refined_dag=_dag(), eda_result=EDAResult())
    state.agent_briefs["eda_agent"] = AgentBrief(
        agent="eda_agent",
        status="done",
        headline="data looks moderately balanced",
        flags=[Flag.TC_IMBALANCE],
        raised_issues=["control group is ~20% smaller than treated"],
    )
    payload = _build_gate_payload(state)
    eda_brief = payload["brief_flags"]["eda_agent"]
    assert eda_brief["status"] == "done"
    assert "tc_imbalance" in eda_brief["flags"]
    assert eda_brief["headline"].startswith("data looks")


def test_gate_payload_skips_missing_briefs_silently():
    state = _state(refined_dag=_dag(), eda_result=EDAResult())
    # No briefs registered at all.
    payload = _build_gate_payload(state)
    assert payload["brief_flags"] == {}


# --- park_for_approval async ----------------------------------------------


@pytest.mark.asyncio
async def test_park_sets_status_emits_event_and_calls_callback():
    state = _state(refined_dag=_dag(), eda_result=EDAResult())

    captured: list[AnalysisState] = []

    async def cb(s: AnalysisState) -> None:
        captured.append(s)

    returned = await park_for_approval(state, status_callback=cb)
    assert returned is state
    assert state.status == JobStatus.AWAITING_APPROVAL
    assert len(captured) == 1 and captured[0] is state
    # SSE event was pushed with the gate payload
    assert any(e["event_type"] == "approval_required" for e in state.sse_events)
    gate_event = next(e for e in state.sse_events if e["event_type"] == "approval_required")
    assert gate_event["data"]["treatment_variable"] == "t"


@pytest.mark.asyncio
async def test_park_without_callback_still_sets_status_and_emits_event():
    state = _state(refined_dag=_dag(), eda_result=EDAResult())
    await park_for_approval(state, status_callback=None)
    assert state.status == JobStatus.AWAITING_APPROVAL
    assert any(e["event_type"] == "approval_required" for e in state.sse_events)
