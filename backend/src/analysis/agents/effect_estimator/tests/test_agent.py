"""Tests for the deterministic fixed-method estimation pass.

_run_fixed_methods replaces the LLM ReAct loop: it runs a fixed method set once
each on the resolved DAG adjustment set, so the forest plot compares genuinely
different methods on one control set instead of OLS re-run on improvised covariate
subsets.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from src.analysis.agents.base import TreatmentEffectResult
from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.effect_estimator.agent import EffectEstimatorAgent

LABELS = {
    "ols": "OLS Regression",
    "ipw": "Inverse Probability Weighting",
    "psm": "Propensity Score Matching",
    "aipw": "Doubly Robust (AIPW)",
}
ESTIMATES = {"ols": 100.0, "ipw": 110.0, "psm": 90.0, "aipw": 95.0}


def _effect(method: str, estimate: float) -> TreatmentEffectResult:
    return TreatmentEffectResult(
        method=method,
        estimand="ATE",
        estimate=estimate,
        std_error=1.0,
        ci_lower=estimate - 1,
        ci_upper=estimate + 1,
        p_value=0.05,
        treatment_variable="treat",
        outcome_variable="re78",
    )


def _agent() -> EffectEstimatorAgent:
    agent = EffectEstimatorAgent()
    agent._treatment_var = "treat"
    agent._outcome_var = "re78"
    agent._covariates = ["age", "educ", "re74", "re75"]
    agent._df = pd.DataFrame({"treat": [0, 1], "re78": [1.0, 2.0]})
    agent._results = []
    return agent


def _state() -> AnalysisState:
    return AnalysisState(
        job_id="t",
        dataset_info=DatasetInfo(url="x"),
        treatment_variable="treat",
        outcome_variable="re78",
    )


def test_runs_each_method_once_on_the_fixed_covariates():
    agent = _agent()
    calls: list[tuple[str, list]] = []

    def fake_run_method(method, t, o, cov, df, st):
        calls.append((method, list(cov)))
        return _effect(LABELS[method], ESTIMATES[method])

    with patch(
        "src.analysis.agents.effect_estimator.agent.run_method",
        side_effect=fake_run_method,
    ):
        agent._run_fixed_methods(_state())

    # one call per fixed method, in order, always on the resolved adjustment set
    assert [m for m, _ in calls] == list(EffectEstimatorAgent.FIXED_METHODS)
    assert all(cov == ["age", "educ", "re74", "re75"] for _, cov in calls)
    # four distinct, distinctly-labeled results, no duplicates
    assert len(agent._results) == 4
    assert {r.method for r in agent._results} == set(LABELS.values())


def test_skips_methods_the_data_cannot_support():
    agent = _agent()

    def fake_run_method(method, t, o, cov, df, st):
        return _effect("OLS Regression", 100.0) if method == "ols" else None

    with patch(
        "src.analysis.agents.effect_estimator.agent.run_method",
        side_effect=fake_run_method,
    ):
        agent._run_fixed_methods(_state())

    assert len(agent._results) == 1
    assert agent._results[0].method == "OLS Regression"


def test_one_method_raising_does_not_sink_the_rest():
    agent = _agent()

    def fake_run_method(method, t, o, cov, df, st):
        if method == "psm":
            raise ValueError("matching blew up")
        return _effect(LABELS[method], ESTIMATES[method])

    with patch(
        "src.analysis.agents.effect_estimator.agent.run_method",
        side_effect=fake_run_method,
    ):
        agent._run_fixed_methods(_state())

    # ols, ipw, aipw succeed; psm raised and was skipped, not fatal
    assert len(agent._results) == 3
    assert "Propensity Score Matching" not in {r.method for r in agent._results}
