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
    LANES,
    LaneInputError,
    matching,
    observational,
    survival,
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
    assert reasons and "complete numeric rows" in reasons[0]
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
