"""DAG gate (checkpoint A): should_pause_for_dag_approval + park_for_dag_approval.

Pins the truth table (pause once dag_expert refines the DAG, before estimation),
that the data-gate approval does NOT pre-satisfy this gate (separate
dag_approval record), and the DAG summary the SSE event carries to the UI.
"""
from __future__ import annotations

import pytest

from src.analysis.agents.base.state import AnalysisState, DatasetInfo, JobStatus
from src.analysis.agents.causal_discovery.output import CausalDAG, CausalEdge
from src.analysis.agents.effect_estimator.output import TreatmentEffectResult
from src.analysis.orchestrator.base import (
    _build_dag_gate_payload,
    park_for_dag_approval,
    should_pause_for_dag_approval,
)
from src.domain.approval import HumanApproval


def _dag(**overrides) -> CausalDAG:
    base = dict(
        nodes=["t", "y", "age"],
        edges=[
            CausalEdge(source="age", target="t"),
            CausalEdge(source="age", target="y"),
            CausalEdge(source="t", target="y"),
        ],
        discovery_method="dag_expert",
        adjustment_set=["age"],
        variable_roles={"age": "confounder"},
    )
    base.update(overrides)
    return CausalDAG(**base)


def _state(**overrides) -> AnalysisState:
    # The DAG gate is always after the data gate, so a realistic pre-DAG-gate
    # state already has the data approved; tests that need it absent override it.
    state = AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(url="kaggle.com/x"),
        treatment_variable="t",
        outcome_variable="y",
        human_approval=HumanApproval.approve(),
    )
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def _effect() -> TreatmentEffectResult:
    return TreatmentEffectResult(method="ols", estimand="ATE", estimate=1.0,
                                 std_error=0.1, ci_lower=0.8, ci_upper=1.2)


# --- should_pause_for_dag_approval truth table -----------------------------


def test_no_refined_dag_does_not_pause():
    assert should_pause_for_dag_approval(_state()) is False


def test_refined_dag_pauses_when_not_yet_dag_approved():
    assert should_pause_for_dag_approval(_state(refined_dag=_dag())) is True


def test_dag_approved_skips_pause():
    state = _state(refined_dag=_dag(), dag_approval=HumanApproval.approve())
    assert should_pause_for_dag_approval(state) is False


def test_data_gate_approval_does_not_satisfy_the_dag_gate():
    # The data-review approval must not pre-approve the DAG: they are separate
    # records, so a refined DAG with only human_approval set still pauses.
    state = _state(refined_dag=_dag(), human_approval=HumanApproval.approve())
    assert should_pause_for_dag_approval(state) is True


def test_dag_rejected_record_still_holds_the_gate():
    state = _state(refined_dag=_dag(), dag_approval=HumanApproval.reject(reason="wrong dag"))
    assert should_pause_for_dag_approval(state) is True


def test_estimation_started_does_not_pause():
    # Effects present means we have resumed past the DAG gate.
    state = _state(refined_dag=_dag(), treatment_effects=[_effect()])
    assert should_pause_for_dag_approval(state) is False


def test_no_pause_before_the_data_gate_is_approved():
    # A refined DAG with the data gate unpassed is not the DAG gate; resume
    # routing must treat this as the data gate, not pause here.
    state = _state(refined_dag=_dag(), human_approval=None)
    assert should_pause_for_dag_approval(state) is False


# --- _build_dag_gate_payload shape -----------------------------------------


def test_dag_payload_marks_the_gate_and_carries_treatment_outcome():
    payload = _build_dag_gate_payload(_state(refined_dag=_dag()))
    assert payload["gate"] == "dag"
    assert payload["treatment_variable"] == "t"
    assert payload["outcome_variable"] == "y"


def test_dag_payload_renders_nodes_edges_and_adjustment_set():
    payload = _build_dag_gate_payload(_state(refined_dag=_dag()))
    dag = payload["dag"]
    assert dag["nodes"] == ["t", "y", "age"]
    assert {"source": "age", "target": "t", "edge_type": "directed"} in dag["edges"]
    assert dag["adjustment_set"] == ["age"]
    assert dag["variable_roles"] == {"age": "confounder"}


def test_dag_payload_is_none_before_any_dag():
    payload = _build_dag_gate_payload(_state())
    assert payload["dag"] is None


# --- park_for_dag_approval async -------------------------------------------


@pytest.mark.asyncio
async def test_park_emits_dag_event_and_calls_callback():
    state = _state(refined_dag=_dag())
    captured: list[AnalysisState] = []

    async def cb(s: AnalysisState) -> None:
        captured.append(s)

    returned = await park_for_dag_approval(state, status_callback=cb)
    assert returned is state
    assert state.status == JobStatus.AWAITING_APPROVAL
    assert len(captured) == 1
    event = next(e for e in state.sse_events if e["event_type"] == "dag_approval_required")
    assert event["data"]["gate"] == "dag"
    assert event["data"]["dag"]["adjustment_set"] == ["age"]
