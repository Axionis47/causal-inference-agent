"""Recipe selection: the right checks for the right design."""
from __future__ import annotations

from src.analysis_v2.agents.targeted_eda.recipes import build_recipe
from src.analysis_v2.spec import CausalSpec, MethodLane, QuestionType, VariableRef


def _names(spec, lane):
    checks, _ = build_recipe(spec, lane)
    return [fn.__name__ for fn in checks]


def test_every_recipe_starts_with_the_base_checks():
    for lane in MethodLane:
        names = _names(CausalSpec(question_type=QuestionType.SIMPLE_EFFECT), lane)
        assert names[:5] == [
            "shape_and_missingness",
            "duplicates",
            "outcome_distribution",
            "exposure_distribution",
            "usable_sample_size",
        ], lane


def test_observational_recipe_runs_balance_and_overlap():
    names = _names(
        CausalSpec(question_type=QuestionType.BINARY_TREATMENT),
        MethodLane.OBSERVATIONAL,
    )
    for expected in ("group_outcome_comparison", "covariate_balance", "propensity_overlap"):
        assert expected in names


def test_did_recipe_runs_trends_and_pre_trends():
    names = _names(CausalSpec(question_type=QuestionType.DID), MethodLane.DID)
    assert "did_trends" in names and "did_pre_trends" in names
    assert "rdd_around_cutoff" not in names


def test_rdd_recipe_runs_cutoff_checks():
    names = _names(CausalSpec(question_type=QuestionType.RDD), MethodLane.RDD)
    assert "rdd_around_cutoff" in names and "rdd_treatment_by_side" in names


def test_question_extras_ride_along():
    interaction = _names(
        CausalSpec(
            question_type=QuestionType.INTERACTION,
            moderator=VariableRef(column="smoker"),
        ),
        MethodLane.OBSERVATIONAL,
    )
    assert "subgroup_outcome_table" in interaction

    multifactor = _names(
        CausalSpec(question_type=QuestionType.MULTI_FACTOR), MethodLane.OBSERVATIONAL
    )
    assert "factor_screening" in multifactor

    dose = _names(
        CausalSpec(question_type=QuestionType.DOSE_RESPONSE, treatment_is_continuous=True),
        MethodLane.OBSERVATIONAL,
    )
    assert "dose_response_scatter" in dose


def test_undecided_lane_still_gets_the_base_recipe():
    checks, targeted = build_recipe(
        CausalSpec(question_type=QuestionType.SIMPLE_EFFECT), None
    )
    assert len(checks) == 5
    assert targeted == []


def test_dag_consistency_runs_only_when_a_graph_exists():
    spec = CausalSpec(question_type=QuestionType.SIMPLE_EFFECT)
    without, _ = build_recipe(spec, MethodLane.OBSERVATIONAL)
    assert "dag_consistency" not in [fn.__name__ for fn in without]
    with_dag, targeted = build_recipe(spec, MethodLane.OBSERVATIONAL, has_dag=True)
    assert "dag_consistency" in [fn.__name__ for fn in with_dag]
    # it is design-agnostic, not a lane-targeted check
    assert "dag_consistency" not in targeted
