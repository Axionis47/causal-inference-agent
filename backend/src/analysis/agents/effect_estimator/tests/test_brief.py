"""Contract tests for the effect_estimator brief.

Pin the capability shape, preflight refusals, and the three flag
derivations (METHOD_UNSTABLE, ATE_OUTLIER, WEAK_INSTRUMENT).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis.agents.base import TreatmentEffectResult
from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.data_profiler.output import DataProfile
from src.analysis.agents.effect_estimator.brief import (
    CAPABILITY,
    build_brief,
    preflight,
)
from src.domain.briefs import Flag


# --- fixtures ---------------------------------------------------------------


def _profile() -> DataProfile:
    return DataProfile(
        n_samples=200,
        n_features=4,
        feature_names=["t", "y", "x1", "x2"],
        feature_types={"t": "binary", "y": "numeric", "x1": "numeric", "x2": "numeric"},
        missing_values={"t": 0, "y": 0, "x1": 0, "x2": 0},
        numeric_stats={},
        categorical_stats={},
    )


def _make_state(
    *,
    dataframe_path: str | None = "/tmp/df.parquet",
    data_profile: DataProfile | None = None,
    treatment_effects: list[TreatmentEffectResult] | None = None,
) -> AnalysisState:
    state = AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://example.com/data"),
        dataframe_path=dataframe_path,
        data_profile=data_profile if data_profile is not None else _profile(),
        treatment_variable="t",
        outcome_variable="y",
    )
    if treatment_effects is not None:
        state.treatment_effects = treatment_effects
    return state


def _effect(
    *,
    method: str = "OLS",
    estimate: float = 1.0,
    details: dict | None = None,
) -> TreatmentEffectResult:
    return TreatmentEffectResult(
        method=method,
        estimand="ATE",
        estimate=estimate,
        std_error=0.1,
        ci_lower=estimate - 0.2,
        ci_upper=estimate + 0.2,
        p_value=0.01,
        treatment_variable="t",
        outcome_variable="y",
        details=details or {},
    )


# --- CAPABILITY -------------------------------------------------------------


class TestCapability:
    def test_name_is_effect_estimator(self):
        assert CAPABILITY.name == "effect_estimator"

    def test_needs_profile_and_dataframe(self):
        assert set(CAPABILITY.needs) == {"data_profile", "dataframe_path"}

    def test_delivers_treatment_effects(self):
        assert "treatment_effects" in CAPABILITY.delivers

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            CAPABILITY.name = "other"

    def test_success_criteria_cover_each_flag(self):
        flags_with_criteria = {
            c.raises_flag for c in CAPABILITY.success_criteria if c.raises_flag
        }
        assert Flag.NEEDS_NOT_MET in flags_with_criteria
        assert Flag.METHOD_UNSTABLE in flags_with_criteria
        assert Flag.ATE_OUTLIER in flags_with_criteria
        assert Flag.WEAK_INSTRUMENT in flags_with_criteria


# --- preflight --------------------------------------------------------------


class TestPreflight:
    def test_refuses_when_data_profile_missing(self):
        state = _make_state()
        state.data_profile = None
        brief = preflight(state)
        assert brief is not None
        assert brief.flags == [Flag.NEEDS_NOT_MET]
        assert "data_profile" in brief.headline

    def test_refuses_when_dataframe_path_missing(self):
        state = _make_state(dataframe_path=None)
        brief = preflight(state)
        assert brief is not None
        assert brief.flags == [Flag.NEEDS_NOT_MET]
        assert "dataframe_path" in brief.headline

    def test_passes_when_needs_satisfied(self):
        state = _make_state()
        assert preflight(state) is None


# --- build_brief: status ----------------------------------------------------


class TestBuildBriefStatus:
    def test_failed_when_no_treatment_effects(self):
        state = _make_state(treatment_effects=[])
        brief = build_brief(state)
        assert brief.status == "failed"
        assert brief.artifact_keys == []
        assert "no treatment effects" in brief.headline.lower()

    def test_done_when_effects_present(self):
        state = _make_state(treatment_effects=[_effect()])
        brief = build_brief(state)
        assert brief.status == "done"
        assert brief.artifact_keys == ["treatment_effects"]


# --- build_brief: METHOD_UNSTABLE ------------------------------------------


class TestBuildBriefMethodUnstable:
    def test_raises_when_cv_exceeds_threshold(self):
        # Estimates 1.0 and 5.0 → mean 3.0, std 2.0, CV ~0.67 > 0.5.
        effects = [
            _effect(method="OLS", estimate=1.0),
            _effect(method="IPW", estimate=5.0),
        ]
        state = _make_state(treatment_effects=effects)
        brief = build_brief(state)
        assert Flag.METHOD_UNSTABLE in brief.flags

    def test_does_not_raise_when_cv_within_threshold(self):
        # Estimates 1.0 and 1.1 → CV ~0.045 << 0.5.
        effects = [
            _effect(method="OLS", estimate=1.0),
            _effect(method="IPW", estimate=1.1),
        ]
        state = _make_state(treatment_effects=effects)
        brief = build_brief(state)
        assert Flag.METHOD_UNSTABLE not in brief.flags

    def test_skipped_with_single_estimate(self):
        # CV undefined; flag should never fire on a singleton.
        state = _make_state(treatment_effects=[_effect()])
        brief = build_brief(state)
        assert Flag.METHOD_UNSTABLE not in brief.flags


# --- build_brief: ATE_OUTLIER ----------------------------------------------


class TestBuildBriefAteOutlier:
    def test_raises_when_one_estimate_far_from_median(self):
        # Cluster at 1.0, outlier at 100.0 → MAD ~0, but the outlier
        # branch only fires when MAD > 0, so make a wider spread.
        effects = [
            _effect(method="OLS", estimate=1.0),
            _effect(method="IPW", estimate=1.2),
            _effect(method="AIPW", estimate=1.4),
            _effect(method="MATCHING", estimate=50.0),
        ]
        state = _make_state(treatment_effects=effects)
        brief = build_brief(state)
        assert Flag.ATE_OUTLIER in brief.flags

    def test_does_not_raise_when_estimates_clustered(self):
        effects = [
            _effect(method="OLS", estimate=1.0),
            _effect(method="IPW", estimate=1.05),
            _effect(method="AIPW", estimate=1.1),
        ]
        state = _make_state(treatment_effects=effects)
        brief = build_brief(state)
        assert Flag.ATE_OUTLIER not in brief.flags


# --- build_brief: WEAK_INSTRUMENT ------------------------------------------


class TestBuildBriefWeakInstrument:
    def test_raises_when_severe_flag_true(self):
        effects = [
            _effect(method="IV", details={"weak_instrument_severe": True}),
        ]
        state = _make_state(treatment_effects=effects)
        brief = build_brief(state)
        assert Flag.WEAK_INSTRUMENT in brief.flags

    def test_raises_when_first_stage_f_below_threshold(self):
        effects = [
            _effect(method="IV", details={"first_stage_f_partial": 4.2}),
        ]
        state = _make_state(treatment_effects=effects)
        brief = build_brief(state)
        assert Flag.WEAK_INSTRUMENT in brief.flags

    def test_does_not_raise_when_strong_instrument(self):
        effects = [
            _effect(method="IV", details={"first_stage_f_partial": 25.0}),
        ]
        state = _make_state(treatment_effects=effects)
        brief = build_brief(state)
        assert Flag.WEAK_INSTRUMENT not in brief.flags

    def test_does_not_raise_for_non_iv_methods(self):
        # No IV-related diagnostics in details; no flag should fire.
        effects = [_effect(method="OLS")]
        state = _make_state(treatment_effects=effects)
        brief = build_brief(state)
        assert Flag.WEAK_INSTRUMENT not in brief.flags
