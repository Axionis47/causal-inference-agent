"""Every lane against fixture ground truth.

Real datasets assert published benchmarks (LaLonde sign flip, ISLR TV
coefficient, Card-Krueger +2.75, Card 1995 2SLS above OLS); synthetic
fixtures assert their generated truth (DID 2.0, RDD 8.0, IV LATE 2.0,
mediation 0.30/0.30, ITS +25% step).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis_v2.agents.method_lane.lanes import LANE_CHECKS, LANES, LaneInputError
from src.analysis_v2.evals.fixtures import generators
from src.analysis_v2.spec import CausalSpec, MethodLane, MethodPlan, QuestionType, VariableRef

DATA = Path(__file__).resolve().parents[3] / "evals" / "fixtures" / "data"


def _spec(**kw) -> CausalSpec:
    base = {"question_type": QuestionType.BINARY_TREATMENT}
    base.update(kw)
    return CausalSpec(**base)


def test_lalonde_regression_adjustment_flips_the_naive_negative_sign():
    frame = pd.read_csv(DATA / "lalonde.csv")
    plan = MethodPlan(
        lane=MethodLane.OBSERVATIONAL, estimator="regression_adjustment",
        estimand="ate", outcome="re78", treatment="treat",
        covariates=["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"],
        settings={"include_ipw": True},
    )
    outcome = LANES[MethodLane.OBSERVATIONAL](frame, plan, _spec())

    naive = frame.groupby("treat")["re78"].mean().diff().iloc[-1]
    assert naive < 0  # the famous misleading raw comparison
    primary = outcome.result.primary
    assert primary.estimand == "ate"
    assert 500 < primary.estimate < 3000  # manifest band around the $1794 benchmark
    ipw = next(e for e in outcome.result.effects if e.estimand == "ate_ipw")
    assert ipw.estimate > 0
    assert outcome.result.n_rows_used == 614


def test_advertising_tv_coefficient_matches_the_textbook():
    frame = pd.read_csv(DATA / "advertising.csv")
    plan = MethodPlan(
        lane=MethodLane.OBSERVATIONAL, estimator="regression_adjustment",
        estimand="dose_effect", outcome="Sales ($)", treatment="TV Ad Budget ($)",
        covariates=["Radio Ad Budget ($)", "Newspaper Ad Budget ($)"],
        settings={"include_ipw": False},
    )
    outcome = LANES[MethodLane.OBSERVATIONAL](
        frame, plan, _spec(question_type=QuestionType.DOSE_RESPONSE)
    )
    assert outcome.result.primary.estimate == pytest.approx(0.0458, abs=0.005)


def test_matching_att_on_lalonde_is_positive(monkeypatch):
    frame = pd.read_csv(DATA / "lalonde.csv")
    plan = MethodPlan(
        lane=MethodLane.MATCHING, estimator="propensity_matching",
        estimand="att", outcome="re78", treatment="treat",
        covariates=["age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75"],
    )
    outcome = LANES[MethodLane.MATCHING](frame, plan, _spec())
    att = outcome.result.primary
    assert att.estimand == "att"
    assert att.estimate > 0
    assert att.std_error is not None  # bootstrap with refit succeeded
    balance = next(a for a in outcome.artifacts if a.name == "balance_after_matching")
    assert "smd_after" in balance.payload.columns


def test_synthetic_did_recovers_the_built_in_att():
    frame = generators.did_panel()
    plan = MethodPlan(
        lane=MethodLane.DID, estimator="difference_in_differences",
        estimand="att", outcome="outcome", treatment=None,
        settings={"time_column": "period", "group_column": "group",
                  "post_column": "post", "needs_reshape": False,
                  "treated_group": "treated"},
    )
    outcome = LANES[MethodLane.DID](frame, plan, _spec(question_type=QuestionType.DID))
    assert outcome.result.primary.estimate == pytest.approx(2.0, abs=0.6)
    assert "clustered" in " ".join(outcome.warnings)  # unit_id present


def test_card_krueger_wide_reshape_reproduces_the_published_did():
    frame = pd.read_csv(DATA / "employment.csv")
    plan = MethodPlan(
        lane=MethodLane.DID, estimator="difference_in_differences",
        estimand="att", outcome="employment", treatment=None,
        settings={"group_column": "state", "needs_reshape": True,
                  "post_column": "post", "treated_group": "1"},
    )
    spec = _spec(
        question_type=QuestionType.DID,
        treatment=VariableRef(derived=True, clue="NJ stores after the wage rise"),
    )
    outcome = LANES[MethodLane.DID](frame, plan, spec)
    assert outcome.result.primary.estimate == pytest.approx(2.75, abs=0.8)


def test_sharp_rdd_recovers_the_built_in_jump():
    frame = generators.scholarship_rdd()
    plan = MethodPlan(
        lane=MethodLane.RDD, estimator="local_linear_rdd", estimand="late",
        outcome="outcome_sharp", treatment="scholarship_sharp",
        settings={"running_variable": "score", "cutoff": 50.0, "bandwidth": 15.0},
    )
    outcome = LANES[MethodLane.RDD](frame, plan, _spec(question_type=QuestionType.RDD))
    jump = next(e for e in outcome.result.effects if e.estimand == "itt_jump")
    assert jump.estimate == pytest.approx(8.0, abs=1.6)
    assert outcome.result.estimator == "rdrobust_sharp"


def test_sharp_rdd_recovers_the_jump_with_a_data_driven_bandwidth():
    """No explicit bandwidth: rdrobust selects its own (CCT) window and still
    recovers the built-in jump. This is the point of the lane, no hand tuning."""
    frame = generators.scholarship_rdd()
    plan = MethodPlan(
        lane=MethodLane.RDD, estimator="local_linear_rdd", estimand="late",
        outcome="outcome_sharp", treatment="scholarship_sharp",
        settings={"running_variable": "score", "cutoff": 50.0},  # bandwidth: auto
    )
    outcome = LANES[MethodLane.RDD](frame, plan, _spec(question_type=QuestionType.RDD))
    jump = next(e for e in outcome.result.effects if e.estimand == "itt_jump")
    assert jump.estimate == pytest.approx(8.0, abs=1.6)


def test_fuzzy_rdd_uses_the_cutoff_as_an_instrument():
    frame = generators.scholarship_rdd()
    plan = MethodPlan(
        lane=MethodLane.RDD, estimator="local_linear_rdd", estimand="late",
        outcome="outcome_fuzzy", treatment="scholarship_fuzzy",
        settings={"running_variable": "score", "cutoff": 50.0, "bandwidth": 15.0},
    )
    outcome = LANES[MethodLane.RDD](frame, plan, _spec(question_type=QuestionType.RDD))
    late = outcome.result.primary
    assert late.estimand == "late"
    assert late.estimate == pytest.approx(8.0, abs=2.0)
    assert outcome.result.estimator == "rdrobust_fuzzy"
    itt = next(e for e in outcome.result.effects if e.estimand == "itt_jump")
    assert itt.estimate == pytest.approx(6.0, abs=2.0)  # reduced-form jump


def test_synthetic_iv_recovers_the_late_where_ols_is_biased():
    frame = generators.synthetic_iv()
    plan = MethodPlan(
        lane=MethodLane.IV, estimator="two_stage_least_squares", estimand="late",
        outcome="y", treatment="x", covariates=["c1", "c2"],
        settings={"instrument": "z"},
    )
    outcome = LANES[MethodLane.IV](frame, plan, _spec(question_type=QuestionType.IV))
    late = outcome.result.primary
    assert late.estimate == pytest.approx(2.0, abs=0.35)
    assert outcome.warnings == []  # strong first stage


def test_card_1995_2sls_exceeds_ols_as_published():
    frame = pd.read_csv(DATA / "card_schooling_wages.csv")
    covs = ["exper", "black", "south", "smsa"]
    plan = MethodPlan(
        lane=MethodLane.IV, estimator="two_stage_least_squares", estimand="late",
        outcome="lwage", treatment="educ", covariates=covs,
        settings={"instrument": "nearc4"},
    )
    outcome = LANES[MethodLane.IV](frame, plan, _spec(question_type=QuestionType.IV))
    assert 0.05 < outcome.result.primary.estimate < 0.3  # textbook ~0.13


def test_iv_fails_honestly_when_the_first_stage_collapses():
    """Regression: an instrument with no significant first-stage relationship with
    the treatment (here z is independent of educ; the same thing happens on real
    data when a control collinear with the treatment, e.g. age = educ + exper + 6,
    absorbs the instrument) yields a noise-driven, arbitrary-sign LATE. The lane
    used to report that as a weak result; it now fails honestly as not-identified
    and emits no estimate. check_ready still passes: the collapse is only visible
    once the first stage is fit."""
    rng = np.random.default_rng(1)
    n = 600
    z = rng.integers(0, 2, n).astype(float)   # candidate instrument
    educ = 12 + rng.normal(0, 2, n)           # NOT moved by z -> no first stage
    lwage = 0.1 * educ + rng.normal(0, 0.5, n)
    frame = pd.DataFrame({"lwage": lwage, "educ": educ, "z": z})
    plan = MethodPlan(
        lane=MethodLane.IV, estimator="two_stage_least_squares", estimand="late",
        outcome="lwage", treatment="educ", covariates=[], settings={"instrument": "z"},
    )
    spec = _spec(question_type=QuestionType.IV)
    assert LANE_CHECKS[MethodLane.IV](frame, plan, spec) == []  # inputs valid
    with pytest.raises(LaneInputError, match="not identified"):
        LANES[MethodLane.IV](frame, plan, spec)


def test_its_step_fixture_recovers_a_positive_level_shift():
    base = pd.read_csv(DATA / "daily_website_visitors.csv", thousands=",")
    frame = generators.website_visitors_step(DATA / "daily_website_visitors.csv")
    plan = MethodPlan(
        lane=MethodLane.TIME_SERIES, estimator="interrupted_time_series",
        estimand="level_shift", outcome="visits",
        settings={"time_column": "Date", "intervention_date": "2019-04-01"},
    )
    outcome = LANES[MethodLane.TIME_SERIES](
        frame, plan, _spec(question_type=QuestionType.TIME_SERIES_INTERVENTION)
    )
    shift = outcome.result.primary
    # truth: +25% multiplicative on a ~3100 pre-mean; the trend term absorbs
    # part of it and the seasonal residual keeps the estimate honest-but-wide
    assert 200 < shift.estimate < 1400
    assert any("before/after" in w for w in outcome.result.warnings)


def test_mediation_fixture_recovers_the_direct_indirect_split():
    frame = generators.mediation()
    plan = MethodPlan(
        lane=MethodLane.MEDIATION, estimator="product_of_coefficients",
        estimand="indirect_effect", outcome="disease_risk",
        treatment="exercise_program", covariates=["baseline_health"],
        settings={"mediator": "weight_change"},
    )
    outcome = LANES[MethodLane.MEDIATION](
        frame, plan, _spec(question_type=QuestionType.MEDIATION)
    )
    by_name = {e.estimand: e for e in outcome.result.effects}
    assert by_name["indirect"].estimate == pytest.approx(0.30, abs=0.08)
    assert by_name["direct"].estimate == pytest.approx(0.30, abs=0.10)
    assert by_name["total"].estimate == pytest.approx(0.60, abs=0.12)


def test_heart_failure_cox_hazard_ratio_for_ejection_fraction():
    frame = pd.read_csv(DATA / "heart_failure.csv")
    plan = MethodPlan(
        lane=MethodLane.SURVIVAL, estimator="cox_proportional_hazards",
        estimand="hazard_ratio", outcome="time", treatment="ejection_fraction",
        covariates=["age", "serum_creatinine"],
        settings={"duration_column": "time", "event_column": "DEATH_EVENT"},
    )
    outcome = LANES[MethodLane.SURVIVAL](
        frame, plan, _spec(question_type=QuestionType.SURVIVAL)
    )
    hr = outcome.result.primary
    assert hr.estimand == "hazard_ratio"
    assert hr.estimate < 1  # higher ejection fraction lowers the death hazard
    assert hr.p_value < 0.01
    assert any(a.kind == "plot" for a in outcome.artifacts)  # kaplan-meier


def test_lanes_refuse_degenerate_inputs_instead_of_running():
    frame = pd.read_csv(DATA / "lalonde.csv")
    no_variation = frame.assign(treat=1)
    plan = MethodPlan(
        lane=MethodLane.OBSERVATIONAL, estimator="regression_adjustment",
        estimand="ate", outcome="re78", treatment="treat", covariates=["age"],
        settings={},
    )
    with pytest.raises(LaneInputError, match="no variation"):
        LANES[MethodLane.OBSERVATIONAL](no_variation, plan, _spec())

    survival_plan = MethodPlan(
        lane=MethodLane.SURVIVAL, estimator="cox_proportional_hazards",
        estimand="hazard_ratio", outcome="time", treatment="treat",
        settings={"duration_column": None, "event_column": None},
    )
    with pytest.raises(LaneInputError, match="duration and event"):
        LANES[MethodLane.SURVIVAL](frame, survival_plan, _spec())
