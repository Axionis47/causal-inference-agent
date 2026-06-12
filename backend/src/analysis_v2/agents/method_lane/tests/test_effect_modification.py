"""The interaction path of the observational lane against ground truth.

Synthetic fixture asserts exact recovery of a constructed interaction;
the insurance fixture asserts the published smoker x bmi modification
(OLS interaction coefficient ~ +1434 per BMI unit, p ~ 1e-129).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis_v2.agents.method_lane.lanes import LANES
from src.analysis_v2.spec import CausalSpec, MethodLane, MethodPlan, QuestionType

DATA = Path(__file__).resolve().parents[3] / "evals" / "fixtures" / "data"


def _plan(moderator: str, **kw) -> MethodPlan:
    base = dict(
        lane=MethodLane.OBSERVATIONAL, estimator="regression_adjustment",
        estimand="ate", settings={"moderator": moderator},
    )
    base.update(kw)
    return MethodPlan(**base)


def _spec() -> CausalSpec:
    return CausalSpec(question_type=QuestionType.INTERACTION)


def test_binary_moderator_interaction_recovers_the_constructed_effects():
    rng = np.random.default_rng(11)
    n = 4000
    t = rng.binomial(1, 0.5, n)
    m = rng.binomial(1, 0.4, n)
    y = 2.0 * t + 1.0 * m + 3.0 * t * m + rng.normal(0, 1, n)
    frame = pd.DataFrame({"t": t, "m": m, "y": np.round(y, 4)})

    outcome = LANES[MethodLane.OBSERVATIONAL](
        frame, _plan("m", outcome="y", treatment="t", covariates=["m"]), _spec()
    )
    by_name = {e.estimand: e for e in outcome.result.effects}
    assert outcome.result.estimator == "interaction_regression"
    assert outcome.result.primary.estimand == "ate"
    assert 2.9 < by_name["interaction"].estimate < 3.1
    assert 1.9 < by_name["cate_low"].estimate < 2.1  # effect at m=0
    assert 4.9 < by_name["cate_high"].estimate < 5.1  # effect at m=1
    assert by_name["interaction"].p_value < 1e-10
    # ate sits at the moderator mean, strictly between the two cates
    assert by_name["cate_low"].estimate < by_name["ate"].estimate < by_name["cate_high"].estimate


def test_insurance_smoker_bmi_modification_matches_the_published_benchmark():
    frame = pd.read_csv(DATA / "insurance.csv")
    frame["smoker"] = (frame["smoker"] == "yes").astype(int)

    outcome = LANES[MethodLane.OBSERVATIONAL](
        frame,
        _plan(
            "bmi", outcome="charges", treatment="smoker",
            covariates=["bmi", "age", "children"],
        ),
        _spec(),
    )
    by_name = {e.estimand: e for e in outcome.result.effects}
    # verified OLS smoker:bmi coefficient ~ +1433.8 per BMI unit
    assert 1100 < by_name["interaction"].estimate < 1800
    assert by_name["interaction"].p_value < 1e-20
    # marginal effect at mean BMI lands in the published $20k-26k range
    assert 19000 < by_name["ate"].estimate < 26000
    # the smoking premium grows materially with BMI (quartile evaluation)
    spread = by_name["cate_high"].estimate - by_name["cate_low"].estimate
    assert spread > 8000
    assert by_name["cate_low"].estimate > 0
    # the model states its own form honestly
    assert any("linear effect-modification" in w for w in outcome.result.warnings)
