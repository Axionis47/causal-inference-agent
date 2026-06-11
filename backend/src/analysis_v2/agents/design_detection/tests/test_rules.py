"""Lane rules: requirements gate eligibility, never the other way."""
from __future__ import annotations

import pandas as pd

from src.analysis_v2.agents.design_detection.rules import (
    detect_wide_periods,
    evaluate_all_lanes,
)
from src.analysis_v2.agents.profiling.tools import build_profile_summary
from src.analysis_v2.spec import (
    CausalSpec,
    EligibilityState,
    MethodLane,
    QuestionType,
    VariableRef,
)


def _profile(df: pd.DataFrame):
    return build_profile_summary(df)


def _lalonde_profile():
    return _profile(
        pd.DataFrame(
            {
                "treat": [0, 1] * 30,
                "re78": [float(i) * 100 for i in range(60)],
                "age": list(range(20, 80)),
            }
        )
    )


def _spec(**overrides) -> CausalSpec:
    base = dict(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(column="treat"),
    )
    base.update(overrides)
    return CausalSpec(**base)


def test_no_lane_with_absent_required_fields_is_ever_enabled():
    """The core invariant: an empty spec enables nothing but observational
    never, and every disabled lane states its missing requirements."""
    bare = CausalSpec(question_type=QuestionType.SIMPLE_EFFECT)
    table = evaluate_all_lanes(bare, _lalonde_profile())
    for entry in table.lanes:
        assert entry.state == EligibilityState.DISABLED, entry.lane
        assert entry.missing_requirements or entry.reason, entry.lane


def test_matching_requires_a_binary_treatment_column():
    profile = _lalonde_profile()
    binary = evaluate_all_lanes(_spec(), profile).for_lane(MethodLane.MATCHING)
    assert binary.state == EligibilityState.ENABLED

    continuous = evaluate_all_lanes(
        _spec(treatment=VariableRef(column="age")), profile
    ).for_lane(MethodLane.MATCHING)
    assert continuous.state == EligibilityState.DISABLED
    assert "not binary" in continuous.reason


def test_rdd_is_conditional_without_a_cutoff_and_enabled_with_one():
    profile = _profile(
        pd.DataFrame(
            {
                "expected_recovery_amount": [float(i) for i in range(100, 200)],
                "actual_recovery_amount": [float(i) for i in range(100)],
            }
        )
    )
    spec = _spec(
        question_type=QuestionType.RDD,
        outcome=VariableRef(column="actual_recovery_amount"),
        treatment=VariableRef(derived=True, clue="recovery strategy above threshold"),
        running_variable=VariableRef(column="expected_recovery_amount"),
    )
    conditional = evaluate_all_lanes(spec, profile).for_lane(MethodLane.RDD)
    assert conditional.state == EligibilityState.CONDITIONAL
    assert "cutoff_value" in conditional.missing_requirements

    enabled = evaluate_all_lanes(
        spec.model_copy(update={"cutoff_value": 1000.0}), profile
    ).for_lane(MethodLane.RDD)
    assert enabled.state == EligibilityState.ENABLED


def test_did_accepts_wide_pre_post_columns_as_conditional():
    profile = _profile(
        pd.DataFrame(
            {
                "state": ["nj", "pa"] * 20,
                "total_emp_feb": [float(i) for i in range(40)],
                "total_emp_nov": [float(i) + 1 for i in range(40)],
            }
        )
    )
    assert set(detect_wide_periods(profile)) == {"total_emp_feb", "total_emp_nov"}
    spec = _spec(
        question_type=QuestionType.DID,
        outcome=VariableRef(candidates=["total_emp_feb", "total_emp_nov"]),
        treatment=VariableRef(derived=True, clue="NJ after the minimum wage rise"),
        group_column=VariableRef(column="state"),
    )
    entry = evaluate_all_lanes(spec, profile).for_lane(MethodLane.DID)
    assert entry.state == EligibilityState.CONDITIONAL
    assert any("reshape" in m for m in entry.missing_requirements)


def test_did_without_group_or_time_is_disabled_with_reasons():
    spec = _spec(question_type=QuestionType.DID, treatment=VariableRef())
    entry = evaluate_all_lanes(spec, _lalonde_profile()).for_lane(MethodLane.DID)
    assert entry.state == EligibilityState.DISABLED
    assert "time structure" in entry.missing_requirements
    assert "treated/control group structure" in entry.missing_requirements


def test_iv_requires_an_explicit_instrument():
    profile = _lalonde_profile()
    no_iv = evaluate_all_lanes(
        _spec(question_type=QuestionType.IV), profile
    ).for_lane(MethodLane.IV)
    assert no_iv.state == EligibilityState.DISABLED

    with_iv = evaluate_all_lanes(
        _spec(question_type=QuestionType.IV, instrument=VariableRef(column="age")),
        profile,
    ).for_lane(MethodLane.IV)
    assert with_iv.state == EligibilityState.ENABLED
    assert "exclusion restriction" in with_iv.reason


def test_survival_needs_both_duration_and_event_columns():
    profile = _profile(
        pd.DataFrame({"time": list(range(60)), "DEATH_EVENT": [0, 1] * 30})
    )
    spec = _spec(
        question_type=QuestionType.SURVIVAL,
        outcome=VariableRef(column="time"),
        duration_column=VariableRef(column="time"),
        event_column=VariableRef(column="DEATH_EVENT"),
    )
    assert (
        evaluate_all_lanes(spec, profile).for_lane(MethodLane.SURVIVAL).state
        == EligibilityState.ENABLED
    )
    missing_event = spec.model_copy(update={"event_column": None})
    entry = evaluate_all_lanes(missing_event, profile).for_lane(MethodLane.SURVIVAL)
    assert entry.state == EligibilityState.DISABLED
    assert "event indicator column" in entry.missing_requirements


def test_time_series_is_conditional_until_the_intervention_date_is_known():
    profile = _profile(
        pd.DataFrame(
            {
                "Date": pd.date_range("2019-01-01", periods=60),
                "visits": [float(i) for i in range(60)],
            }
        )
    )
    spec = _spec(
        question_type=QuestionType.TIME_SERIES_INTERVENTION,
        outcome=VariableRef(column="visits"),
        treatment=VariableRef(derived=True, clue="campaign launch"),
        time_column=VariableRef(column="Date"),
    )
    entry = evaluate_all_lanes(spec, profile).for_lane(MethodLane.TIME_SERIES)
    assert entry.state == EligibilityState.CONDITIONAL
    assert "intervention_date" in entry.missing_requirements

    dated = spec.model_copy(update={"intervention_date": "2019-04-01"})
    assert (
        evaluate_all_lanes(dated, profile).for_lane(MethodLane.TIME_SERIES).state
        == EligibilityState.ENABLED
    )
