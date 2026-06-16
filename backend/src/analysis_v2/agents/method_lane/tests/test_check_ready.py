"""check_ready mirrors each lane's runtime guards exactly.

The readiness checker (S6b) will call check_ready; the lane's run() raises the
first reason it returns. These tests pin the load-bearing property: the gate and
the lane agree on the SAME blocking condition, so a plan check_ready calls clean
can never blow up at S7. They would fail the moment the two paths drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis_v2.agents.method_lane.lanes import (
    LANE_CHECKS,
    LANES,
    LaneInputError,
    did,
    matching,
    observational,
    rdd,
    survival,
    time_series,
)
from src.analysis_v2.spec import CausalSpec, MethodLane, MethodPlan, QuestionType


def _spec() -> CausalSpec:
    return CausalSpec(question_type=QuestionType.BINARY_TREATMENT)


def _obs_plan(**kw) -> MethodPlan:
    base = {
        "lane": MethodLane.OBSERVATIONAL, "estimator": "regression_adjustment",
        "estimand": "ate", "outcome": "y", "treatment": "t",
        "covariates": ["x"], "settings": {},
    }
    base.update(kw)
    return MethodPlan(**base)


def _obs_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"y": rng.normal(size=n), "t": rng.integers(0, 2, n), "x": rng.normal(size=n)}
    )


def test_observational_is_ready_on_a_clean_frame():
    assert observational.check_ready(_obs_frame(), _obs_plan(), _spec()) == []


def test_observational_flags_too_few_rows_and_the_lane_raises_the_same():
    frame = _obs_frame(10)
    reasons = observational.check_ready(frame, _obs_plan(), _spec())
    assert reasons and "complete rows across the design" in reasons[0]
    with pytest.raises(LaneInputError) as exc:
        LANES[MethodLane.OBSERVATIONAL](frame, _obs_plan(), _spec())
    assert str(exc.value) == reasons[0]  # gate and lane agree exactly


def test_observational_flags_a_constant_treatment():
    frame = _obs_frame()
    frame["t"] = 1
    assert observational.check_ready(frame, _obs_plan(), _spec()) == [
        "observational: treatment 't' has no variation"
    ]


def test_matching_flags_a_thin_arm_and_the_lane_raises_the_same():
    rng = np.random.default_rng(1)
    n = 60
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "t": np.array([1] * 5 + [0] * 55),  # only 5 treated -> < 10 per arm
            "x": rng.normal(size=n),
        }
    )
    plan = MethodPlan(
        lane=MethodLane.MATCHING, estimator="propensity_matching",
        estimand="att", outcome="y", treatment="t", covariates=["x"],
    )
    reasons = matching.check_ready(frame, plan, _spec())
    assert reasons == ["matching: needs at least 10 units in each arm"]
    with pytest.raises(LaneInputError) as exc:
        LANES[MethodLane.MATCHING](frame, plan, _spec())
    assert str(exc.value) == reasons[0]


def test_survival_flags_too_few_events_and_the_lane_raises_the_same():
    rng = np.random.default_rng(2)
    n = 60
    frame = pd.DataFrame(
        {
            "time": rng.uniform(1, 100, n),
            "DEATH": np.array([1] * 4 + [0] * 56),  # only 4 events
            "t": rng.integers(0, 2, n),
        }
    )
    plan = MethodPlan(
        lane=MethodLane.SURVIVAL, estimator="cox_proportional_hazards",
        estimand="hazard_ratio", outcome="time", treatment="t",
        settings={"duration_column": "time", "event_column": "DEATH"},
    )
    reasons = survival.check_ready(frame, plan, _spec())
    assert reasons == ["survival: fewer than 10 events observed"]
    with pytest.raises(LaneInputError) as exc:
        LANES[MethodLane.SURVIVAL](frame, plan, _spec())
    assert str(exc.value) == reasons[0]


def test_did_flags_a_single_group_and_the_lane_raises_the_same():
    frame = pd.DataFrame(
        {"y": np.arange(40, dtype=float), "state": ["NJ"] * 40, "period": [0, 1] * 20}
    )
    plan = MethodPlan(
        lane=MethodLane.DID, estimator="difference_in_differences",
        estimand="att", outcome="y", treatment=None,
        settings={"group_column": "state", "time_column": "period"},
    )
    reasons = did.check_ready(frame, plan, _spec())
    assert reasons == ["did: needs exactly two groups, found 1"]
    with pytest.raises(LaneInputError) as exc:  # the _prepare refactor preserved this
        LANES[MethodLane.DID](frame, plan, _spec())
    assert str(exc.value) == reasons[0]


def test_rdd_flags_too_few_rows_in_the_bandwidth():
    frame = pd.DataFrame(
        {"y": np.arange(200, dtype=float), "run": np.arange(200, dtype=float)}
    )
    plan = MethodPlan(
        lane=MethodLane.RDD, estimator="local_linear_rdd", estimand="late",
        outcome="y", treatment=None,
        settings={"running_variable": "run", "cutoff": 100, "bandwidth": 5},
    )
    reasons = rdd.check_ready(frame, plan, _spec())
    assert reasons and "within bandwidth" in reasons[0]


def test_time_series_flags_too_few_points():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    frame = pd.DataFrame({"date": dates, "y": np.arange(20, dtype=float)})
    plan = MethodPlan(
        lane=MethodLane.TIME_SERIES, estimator="interrupted_time_series",
        estimand="level_shift", outcome="y", treatment=None,
        settings={"time_column": "date", "intervention_date": "2020-01-10"},
    )
    assert time_series.check_ready(frame, plan, _spec()) == [
        "time_series: only 20 time points"
    ]


def test_mediation_tolerates_categorical_covariates_and_runs():
    """Regression: categorical confounders must be one-hot encoded, not coerced
    to NaN. Before the fix the whole adjustment set went through numeric
    coercion, so any string covariate emptied the complete-case frame, check_ready
    reported '0 complete rows', and select_runnable silently demoted mediation to
    the observational lane (no indirect/direct decomposition ever produced)."""
    rng = np.random.default_rng(3)
    n = 120
    t = rng.integers(1, 5, n).astype(float)         # ordinal, studytime-like
    m = 0.5 * t + rng.normal(size=n)                # mediator, early-grade-like
    y = 0.8 * m + 0.2 * t + rng.normal(size=n)      # outcome, final-grade-like
    frame = pd.DataFrame(
        {
            "G3": y, "studytime": t, "G1": m,
            "Mjob": rng.choice(["teacher", "services", "at_home"], n),  # categorical
            "school": rng.choice(["GP", "MS"], n),                      # categorical
        }
    )
    plan = MethodPlan(
        lane=MethodLane.MEDIATION, estimator="product_of_coefficients",
        estimand="indirect", outcome="G3", treatment="studytime",
        covariates=["Mjob", "school"], settings={"mediator": "G1"},
    )
    # the gate no longer blocks on the string covariates
    assert LANE_CHECKS[MethodLane.MEDIATION](frame, plan, _spec()) == []
    # and the lane runs, producing the decomposition the user asked for
    out = LANES[MethodLane.MEDIATION](frame, plan, _spec())
    assert out.result.lane == MethodLane.MEDIATION
    assert {"indirect", "direct", "total"} <= {e.estimand for e in out.result.effects}


def test_lane_checks_covers_every_lane():
    assert set(LANE_CHECKS) == set(LANES)
