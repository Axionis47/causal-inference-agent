"""Tests for critique helpers, focused on the adjustment-aware REJECT guard.

On observational data the raw treatment/control groups are imbalanced; that is
the reason to adjust, not grounds to fail the analysis. The guard overrides a
REJECT founded on raw imbalance when the analysis actually adjusted for a
non-empty confounder set.
"""
from src.analysis.agents.base import (
    AnalysisState,
    CausalDAG,
    CritiqueDecision,
    DatasetInfo,
)
from src.analysis.agents.critique.helpers import create_feedback
from src.analysis.agents.effect_estimator.output import TreatmentEffectResult


def _state(adjustment, n_methods):
    state = AnalysisState(
        job_id="c",
        dataset_info=DatasetInfo(url="x"),
        treatment_variable="treat",
        outcome_variable="re78",
    )
    state.refined_dag = CausalDAG(
        nodes=["treat", "re78", *adjustment],
        edges=[],
        discovery_method="test",
        adjustment_set=list(adjustment),
    )
    state.treatment_effects = [
        TreatmentEffectResult(
            method="Doubly Robust (AIPW)",
            estimand="ATE",
            estimate=1700.0,
            std_error=600.0,
            ci_lower=500.0,
            ci_upper=2900.0,
            p_value=0.01,
        )
        for _ in range(n_methods)
    ]
    return state


_REJECT_ON_IMBALANCE = {
    "decision": "REJECT",
    "reasoning": "Severe covariate imbalance across 7 of 8 covariates leads to confounding.",
    "issues": ["Severe covariate imbalance detected (SMD > 0.25)."],
    "scores": {"statistical_validity": 1},
}


def test_reject_on_raw_imbalance_is_overridden_when_adjusted():
    state = _state(adjustment=["age", "educ", "re74", "re75"], n_methods=6)
    fb = create_feedback(_REJECT_ON_IMBALANCE, iteration=1, state=state)
    assert fb.decision == CritiqueDecision.APPROVE
    assert "adjustment-aware override" in fb.reasoning


def test_reject_stands_when_no_adjustment_set():
    state = _state(adjustment=[], n_methods=1)
    fb = create_feedback(_REJECT_ON_IMBALANCE, iteration=1, state=state)
    assert fb.decision == CritiqueDecision.REJECT


def test_reject_for_other_reasons_is_not_overridden():
    state = _state(adjustment=["age", "educ"], n_methods=6)
    other = {
        "decision": "REJECT",
        "reasoning": "Estimates are wildly inconsistent across methods and unstable.",
        "issues": ["Estimates vary by 10x across methods."],
    }
    fb = create_feedback(other, iteration=1, state=state)
    assert fb.decision == CritiqueDecision.REJECT


def test_approve_is_passed_through_unchanged():
    state = _state(adjustment=["age"], n_methods=3)
    fb = create_feedback({"decision": "APPROVE", "reasoning": "Looks good."}, 1, state)
    assert fb.decision == CritiqueDecision.APPROVE


def test_guard_is_skipped_without_state():
    fb = create_feedback(_REJECT_ON_IMBALANCE, iteration=1)
    assert fb.decision == CritiqueDecision.REJECT
