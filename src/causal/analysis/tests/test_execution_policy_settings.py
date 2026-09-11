"""Dispatch regression captured before moving constants into method-owned policies."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from causal.analysis.common.catalog import method_module
from causal.analysis.integration.numerical import _settings, prepare
from causal.analysis.tests.support.interface import approve, design


@pytest.mark.parametrize("method", ["randomized", "aipw", "did", "rdd"])
def test_numerical_dispatch_remains_identical_to_previous_implementation(method: str) -> None:
    snapshot = json.loads(Path(__file__).with_name("support").joinpath(
        "fixed_execution_settings.json").read_text())[method]
    configuration = method_module(method).Specification(**snapshot["configuration"])
    plan = SimpleNamespace(specification=SimpleNamespace(
        configuration=configuration, design=SimpleNamespace(outcome=SimpleNamespace(column="y"))))
    actual = _settings(plan)
    assert json.loads(json.dumps(actual)) == snapshot["settings"]
    definition = method_module(method).DEFINITION
    assert definition.fixed_policies
    assert all(policy.description for policy in definition.fixed_policies)
    assert len({policy.id for policy in definition.fixed_policies}) == len(definition.fixed_policies)


def test_aipw_retains_the_exact_nuisance_profile_and_randomness_policies() -> None:
    data = pl.DataFrame({"id": list(range(24)), "arm": ["control", "treated"] * 12,
                         "x": [i / 10 for i in range(24)], "y": [float(i % 7) for i in range(24)]})
    approved = approve(data, design("aipw", {"adjustment_set_pre_treatment": True}), {
        "method": "aipw", "estimand": "ate", "treatment_column": "arm", "unit_column": "id",
        "treated_value": "treated", "comparator_value": "control", "covariate_columns": ("x",)})
    context = prepare(approved.plan, data)
    profile = context.pack.nuisance_profiles[0]
    assert profile.profile_id == "regularized_glm" and profile.role == "primary"
    assert profile.propensity_learner == "sklearn_logistic_regression_l2"
    assert profile.outcome_learner == "sklearn_ridge_regression_l2"
    assert profile.hyperparameters == {
        "regularization_c": 1.0, "max_iter": 1000, "standardize": True, "random_state": 0}
    assert context.legacy_plan.seed == 17
    assert context.legacy_plan.fold_assignment_rule_id == "stratified_by_treatment_and_unit"
