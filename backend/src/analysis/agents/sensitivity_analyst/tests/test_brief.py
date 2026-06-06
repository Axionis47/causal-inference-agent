"""Contract tests for the sensitivity_analyst brief.

Pin the capability shape, preflight refusal, and flag derivation.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.base import SensitivityResult, TreatmentEffectResult
from src.analysis.agents.sensitivity_analyst.brief import (
    CAPABILITY,
    build_brief,
    preflight,
)
from src.domain.briefs import Flag


# --- fixtures ---------------------------------------------------------------


def _effect(estimate: float = 1.0) -> TreatmentEffectResult:
    return TreatmentEffectResult(
        method="OLS",
        estimand="ATE",
        estimate=estimate,
        std_error=0.1,
        ci_lower=estimate - 0.2,
        ci_upper=estimate + 0.2,
        p_value=0.01,
        treatment_variable="t",
        outcome_variable="y",
    )


def _make_state(
    *,
    treatment_effects: list[TreatmentEffectResult] | None = None,
    sensitivity_results: list[SensitivityResult] | None = None,
) -> AnalysisState:
    state = AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://example.com/data"),
        treatment_variable="t",
        outcome_variable="y",
    )
    if treatment_effects is not None:
        state.treatment_effects = treatment_effects
    if sensitivity_results is not None:
        state.sensitivity_results = sensitivity_results
    return state


def _sens(
    method: str,
    robustness_value: float = 2.0,
    interpretation: str = "ROBUST",
    details: dict | None = None,
) -> SensitivityResult:
    return SensitivityResult(
        method=method,
        robustness_value=robustness_value,
        interpretation=interpretation,
        details=details or {},
    )


# --- CAPABILITY -------------------------------------------------------------


class TestCapability:
    def test_name_is_sensitivity_analyst(self):
        assert CAPABILITY.name == "sensitivity_analyst"

    def test_needs_treatment_effects(self):
        assert CAPABILITY.needs == ("treatment_effects",)

    def test_delivers_sensitivity_results(self):
        assert "sensitivity_results" in CAPABILITY.delivers

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            CAPABILITY.name = "other"

    def test_success_criteria_cover_each_flag(self):
        flags_with_criteria = {
            c.raises_flag for c in CAPABILITY.success_criteria if c.raises_flag
        }
        assert Flag.NEEDS_NOT_MET in flags_with_criteria
        assert Flag.WEAK_TO_UNOBSERVED in flags_with_criteria
        assert Flag.NOT_ROBUST in flags_with_criteria


# --- preflight --------------------------------------------------------------


class TestPreflight:
    def test_refuses_when_treatment_effects_empty(self):
        state = _make_state(treatment_effects=[])
        brief = preflight(state)
        assert brief is not None
        assert brief.flags == [Flag.NEEDS_NOT_MET]
        assert "treatment_effects" in brief.headline

    def test_passes_when_treatment_effects_present(self):
        state = _make_state(treatment_effects=[_effect()])
        assert preflight(state) is None


# --- build_brief: status ----------------------------------------------------


class TestBuildBriefStatus:
    def test_failed_when_no_sensitivity_results(self):
        state = _make_state(treatment_effects=[_effect()], sensitivity_results=[])
        brief = build_brief(state)
        assert brief.status == "failed"
        assert brief.artifact_keys == []
        assert "no results" in brief.headline.lower()

    def test_done_when_results_present(self):
        results = [_sens("E-value", robustness_value=3.0)]
        state = _make_state(
            treatment_effects=[_effect()], sensitivity_results=results
        )
        brief = build_brief(state)
        assert brief.status == "done"
        assert brief.artifact_keys == ["sensitivity_results"]


# --- build_brief: WEAK_TO_UNOBSERVED ---------------------------------------


class TestBuildBriefWeakToUnobserved:
    def test_raises_when_e_value_below_threshold(self):
        results = [_sens("E-value", robustness_value=1.2)]
        state = _make_state(
            treatment_effects=[_effect()], sensitivity_results=results
        )
        brief = build_brief(state)
        assert Flag.WEAK_TO_UNOBSERVED in brief.flags

    def test_does_not_raise_at_exactly_threshold(self):
        results = [_sens("E-value", robustness_value=1.5)]
        state = _make_state(
            treatment_effects=[_effect()], sensitivity_results=results
        )
        brief = build_brief(state)
        assert Flag.WEAK_TO_UNOBSERVED not in brief.flags

    def test_does_not_raise_when_no_e_value_method(self):
        results = [_sens("Rosenbaum Bounds (approximate)", robustness_value=1.0)]
        state = _make_state(
            treatment_effects=[_effect()], sensitivity_results=results
        )
        brief = build_brief(state)
        assert Flag.WEAK_TO_UNOBSERVED not in brief.flags


# --- build_brief: NOT_ROBUST -----------------------------------------------


class TestBuildBriefNotRobust:
    def test_raises_when_two_methods_concerned(self):
        results = [
            _sens("E-value", robustness_value=1.0),
            _sens(
                "Specification Curve",
                robustness_value=0.3,
                details={"cv": 0.6, "all_same_sign": True},
            ),
        ]
        state = _make_state(
            treatment_effects=[_effect()], sensitivity_results=results
        )
        brief = build_brief(state)
        assert Flag.NOT_ROBUST in brief.flags

    def test_does_not_raise_with_only_one_concern(self):
        # E-value concern alone fires WEAK_TO_UNOBSERVED but not NOT_ROBUST.
        results = [
            _sens("E-value", robustness_value=1.0),
            _sens("Bootstrap Variance Check", interpretation="STABLE: fine"),
        ]
        state = _make_state(
            treatment_effects=[_effect()], sensitivity_results=results
        )
        brief = build_brief(state)
        assert Flag.NOT_ROBUST not in brief.flags

    def test_subgroup_heterogeneous_counts_as_concern(self):
        results = [
            _sens("E-value", robustness_value=1.0),
            _sens(
                "Subgroup Analysis",
                interpretation="HETEROGENEOUS: Effect varies including sign changes",
            ),
        ]
        state = _make_state(
            treatment_effects=[_effect()], sensitivity_results=results
        )
        brief = build_brief(state)
        assert Flag.NOT_ROBUST in brief.flags

    def test_placebo_concerning_counts_as_concern(self):
        results = [
            _sens(
                "Placebo Test (both)",
                interpretation="CONCERNING: Real effect within placebo distribution",
            ),
            _sens(
                "Bootstrap Variance Check",
                interpretation="UNSTABLE: Bootstrap SE larger than reported",
            ),
        ]
        state = _make_state(
            treatment_effects=[_effect()], sensitivity_results=results
        )
        brief = build_brief(state)
        assert Flag.NOT_ROBUST in brief.flags

    def test_spec_curve_sign_change_counts_as_concern(self):
        results = [
            _sens(
                "Specification Curve",
                details={"cv": 0.1, "all_same_sign": False},
            ),
            _sens("Rosenbaum Bounds (approximate)", robustness_value=1.0),
        ]
        state = _make_state(
            treatment_effects=[_effect()], sensitivity_results=results
        )
        brief = build_brief(state)
        assert Flag.NOT_ROBUST in brief.flags
