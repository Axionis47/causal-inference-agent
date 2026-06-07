"""Results gate (checkpoint B): should_pause_for_results + park_for_results_approval.

Pins the truth table (pause once estimates and sensitivity exist, before
critique), that an earlier approval does not pre-satisfy it, and the effects /
sensitivity / balance the SSE event carries so the frontend can plot it.
"""
from __future__ import annotations

import pytest

from src.analysis.agents.base.state import AnalysisState, DatasetInfo, JobStatus
from src.analysis.agents.eda.output import EDAResult
from src.analysis.agents.effect_estimator.output import TreatmentEffectResult
from src.analysis.agents.sensitivity_analyst.output import SensitivityResult
from src.analysis.orchestrator.base import (
    _build_results_gate_payload,
    park_for_results_approval,
    should_pause_for_results,
)
from src.domain.approval import HumanApproval


def _effect(method="OLS", est=1.0) -> TreatmentEffectResult:
    return TreatmentEffectResult(method=method, estimand="ATE", estimate=est,
                                 std_error=0.1, ci_lower=est - 0.2, ci_upper=est + 0.2, p_value=0.03)


def _sens() -> SensitivityResult:
    return SensitivityResult(method="evalue", robustness_value=1.8, interpretation="moderately robust")


def _state(**overrides) -> AnalysisState:
    # The results gate is always after the DAG gate, so a realistic pre-results
    # state has the data and DAG already approved.
    state = AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(url="kaggle.com/x"),
        treatment_variable="t",
        outcome_variable="y",
        human_approval=HumanApproval.approve(),
        dag_approval=HumanApproval.approve(),
    )
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


# --- should_pause_for_results truth table ----------------------------------


def test_no_effects_does_not_pause():
    assert should_pause_for_results(_state()) is False


def test_effects_without_sensitivity_does_not_pause():
    assert should_pause_for_results(_state(treatment_effects=[_effect()])) is False


def test_effects_and_sensitivity_pause_when_not_yet_results_approved():
    state = _state(treatment_effects=[_effect()], sensitivity_results=[_sens()])
    assert should_pause_for_results(state) is True


def test_results_approved_skips_pause():
    state = _state(treatment_effects=[_effect()], sensitivity_results=[_sens()],
                   results_approval=HumanApproval.approve())
    assert should_pause_for_results(state) is False


def test_does_not_pause_before_the_dag_gate_is_approved():
    state = _state(treatment_effects=[_effect()], sensitivity_results=[_sens()], dag_approval=None)
    assert should_pause_for_results(state) is False


def test_results_rejected_record_still_holds_the_gate():
    state = _state(treatment_effects=[_effect()], sensitivity_results=[_sens()],
                   results_approval=HumanApproval.reject(reason="implausible effect size"))
    assert should_pause_for_results(state) is True


# --- _build_results_gate_payload shape -------------------------------------


def test_payload_marks_the_gate_and_carries_the_effects_forest_data():
    state = _state(treatment_effects=[_effect("OLS", 1.0), _effect("IPW", 1.3)],
                   sensitivity_results=[_sens()])
    payload = _build_results_gate_payload(state)
    assert payload["gate"] == "results"
    methods = [e["method"] for e in payload["effects"]]
    assert methods == ["OLS", "IPW"]
    assert payload["effects"][0]["ci_lower"] == pytest.approx(0.8)
    assert payload["effects"][0]["ci_upper"] == pytest.approx(1.2)


def test_payload_carries_sensitivity_and_balance_for_the_love_plot():
    state = _state(
        treatment_effects=[_effect()],
        sensitivity_results=[_sens()],
        eda_result=EDAResult(covariate_balance={"age": {"smd": 0.3}, "educ": {"smd": 0.05}}),
    )
    payload = _build_results_gate_payload(state)
    assert payload["sensitivity"][0]["robustness_value"] == 1.8
    balance = {b["covariate"]: b["smd"] for b in payload["balance"]}
    assert balance == {"age": 0.3, "educ": 0.05}


# --- park_for_results_approval async ---------------------------------------


@pytest.mark.asyncio
async def test_park_emits_results_event_and_calls_callback():
    state = _state(treatment_effects=[_effect()], sensitivity_results=[_sens()])
    captured: list[AnalysisState] = []

    async def cb(s: AnalysisState) -> None:
        captured.append(s)

    await park_for_results_approval(state, status_callback=cb)
    assert state.status == JobStatus.AWAITING_APPROVAL
    assert len(captured) == 1
    event = next(e for e in state.sse_events if e["event_type"] == "results_approval_required")
    assert event["data"]["gate"] == "results"
    assert event["data"]["effects"][0]["estimate"] == 1.0
