"""RCT adapter regression for source-shaped aliased categorical randomization strata."""

from __future__ import annotations

import math

import pandas as pd
import polars as pl
import pytest
from pyfixest.estimation import feols

from causal.analysis.methods.randomized import estimation as rct
from causal.analysis.tests.support.builders import PACK, make_plan, ref

ROLES = {"treatment": "treatment", "outcome": "turnout_rate",
         "stratum": "randomization_stratum", "precision_covariate": "randomization_stratum",
         "cluster": "cable_system_id", "unit_identifier": "cable_system_id"}


def trial() -> pl.DataFrame:
    # Rock the Vote's source shape: 35 two-system and five three-system strata, 85 unique
    # assignment clusters, and numeric-looking IDs shared by stratum and precision roles.
    rows = []
    for stratum in range(1, 41):
        arms = (0, 1) if stratum <= 35 else (0, 1, 1) if stratum <= 37 else (0, 0, 1)
        for arm in arms:
            uid = len(rows) + 1
            rows.append({"cable_system_id": uid, "treatment": arm,
                         "randomization_stratum": stratum,
                         "turnout_rate": 0.5 + 0.02 * arm + stratum / 1000 + math.sin(uid) / 100})
    return pl.DataFrame(rows)


def test_aliased_numeric_stratum_is_cast_once_as_a_category_and_absorbed_once() -> None:
    source = trial()
    assert source.height == 85 and source.schema["randomization_stratum"] == pl.Int64
    frame = rct._typed(source, ROLES)
    assert frame.height == source.height and frame["randomization_stratum"].n_unique() == 40
    assert frame.schema["randomization_stratum"] == pl.String
    assert frame.schema["turnout_rate"] == pl.Float64
    params = dict(PACK.parameter_defaults)
    assert rct._formula(ROLES, params) == "turnout_rate ~ treated | randomization_stratum"
    assert rct._formula(ROLES, params | {"stratum_handling": "none"}) == (
        "turnout_rate ~ treated + randomization_stratum")
    distinct = ROLES | {"precision_covariate": "baseline_turnout"}
    assert rct._formula(distinct, params) == (
        "turnout_rate ~ treated + baseline_turnout | randomization_stratum")


def test_source_shaped_aliased_rct_matches_direct_stratum_fixed_effects_crv1() -> None:
    source = trial()
    plan = make_plan(role_columns=ROLES, contrast_ids=("1_vs_0",),
                     estimator_parameters=dict(PACK.parameter_defaults))
    actual = rct.RandomizedExperimentAdapter(ref("mask")).fit(source, plan, PACK)
    item = actual.items[0]
    data = pd.DataFrame({name: source[name].to_list() for name in source.columns})
    data["treated"] = data["treatment"].astype(float)
    data["randomization_stratum"] = data["randomization_stratum"].astype(str)
    data["cable_system_id"] = data["cable_system_id"].astype(str)
    direct = feols("turnout_rate ~ treated | randomization_stratum", data=data,
                   vcov={"CRV1": "cable_system_id"})
    assert item.estimate == pytest.approx(float(direct.coef()["treated"]), abs=1e-14)
    assert item.standard_error == pytest.approx(float(direct.se()["treated"]), abs=1e-14)
    assert (item.interval_lower, item.interval_upper) == pytest.approx(
        tuple(direct.confint(alpha=0.05).loc["treated"]), abs=1e-14)
    assert item.contributing_counts == {"row": 85, "unit": 85}
    balance = actual.harvest["baseline_balance"]
    assert len([name for name in balance if name.endswith(":standardized_difference")]) == 40
    assert item.method_quantities["fitted_covariance"] == "{'CRV1': 'cable_system_id'}"


def test_truthful_cr1_default_preserves_legacy_covariance_numbers() -> None:
    assert PACK.parameter_defaults["cluster_covariance"] == "cluster_robust_cr1"
    assert PACK.finite_sample_correction == "hc2_cr1"
    plan = make_plan(role_columns=ROLES, contrast_ids=("1_vs_0",),
                     estimator_parameters=dict(PACK.parameter_defaults))
    adapter = rct.RandomizedExperimentAdapter(ref("mask"))
    item = adapter.fit(trial(), plan, PACK).items[0]
    legacy = plan.model_copy(update={"finite_sample_correction": "hc2_cr2",
        "estimator_parameters": dict(plan.estimator_parameters) | {"cluster_covariance": "cluster_robust_cr2"}})
    old = adapter.fit(trial(), legacy, PACK).items[0]
    for name in ("estimate", "standard_error", "interval_lower", "interval_upper", "p_value"):
        assert getattr(item, name) == getattr(old, name)
    assert item.finite_sample_correction == old.finite_sample_correction == "hc2_cr1"
    assert old.method_quantities["covariance_profile"] == "cluster_robust_cr1"
    assert old.method_quantities["requested_covariance_profile"] == "cluster_robust_cr2"
    assert rct._vcov({}, plan.estimator_parameters) == rct._vcov({}, legacy.estimator_parameters) == "HC2"


@pytest.mark.parametrize(("estimator", "adjust", "stratify", "cluster"), [
    ("difference_in_means", False, False, False),
    ("ancova", True, False, False),
    ("difference_in_means", False, True, True),
])
def test_public_configuration_reaches_the_declared_formula_and_covariance(
        estimator: str, adjust: bool, stratify: bool, cluster: bool) -> None:
    from causal.analysis import interface as api
    from causal.analysis.tests.support.interface import approve, design

    source = trial()
    fixed = design("randomized", {"assignment_mechanism": (
        "cluster_randomized" if cluster else "individual_randomized")}, outcome="turnout_rate")
    configuration = {
        "method": "randomized", "treatment_column": "treatment",
        "unit_column": "cable_system_id", "treated_value": "1", "comparator_value": "0",
        "estimator": estimator,
        "precision_covariate_column": "randomization_stratum" if adjust else None,
        "stratum_column": "randomization_stratum" if stratify else None,
        "cluster_column": "cable_system_id" if cluster else None}
    actual = api.execute(approve(source, fixed, configuration), source)
    data = pd.DataFrame({name: source[name].to_list() for name in source.columns})
    data["treated"] = data["treatment"].astype(float)
    formula = "turnout_rate ~ treated" + (" + randomization_stratum" if adjust else "")
    if stratify:
        data["randomization_stratum"] = data["randomization_stratum"].astype(str)
        formula += " | randomization_stratum"
    direct = feols(formula, data=data, vcov={"CRV1": "cable_system_id"} if cluster else "HC2")
    assert actual.primary.status == "computed", actual.primary.explanation
    item = actual.primary.estimates[0]
    assert item.estimate == pytest.approx(float(direct.coef()["treated"]), abs=1e-12)
    assert item.standard_error == pytest.approx(float(direct.se()["treated"]), abs=1e-12)


def test_public_ancova_sensitivity_executes_the_unadjusted_contrast() -> None:
    from causal.analysis import interface as api
    from causal.analysis.tests.support.interface import approve, design

    source = trial()
    fixed = design("randomized", {"assignment_mechanism": "individual_randomized"},
                   outcome="turnout_rate")
    configuration = {"method": "randomized", "treatment_column": "treatment",
                     "unit_column": "cable_system_id", "treated_value": "1", "comparator_value": "0",
                     "estimator": "ancova", "precision_covariate_column": "randomization_stratum"}
    actual = api.execute(approve(source, fixed, configuration,
                                 sensitivities=("unadjusted_itt",)), source)
    branch = actual.sensitivities[0]
    difference = source.filter(pl.col("treatment") == 1)["turnout_rate"].mean() - source.filter(
        pl.col("treatment") == 0)["turnout_rate"].mean()
    assert branch.computation_id == "unadjusted_itt" and branch.status == "computed"
    assert branch.estimates[0].estimate == pytest.approx(difference, abs=1e-12)
    assert actual.primary.estimates[0].estimate != pytest.approx(branch.estimates[0].estimate, abs=1e-5)


def test_public_continuous_outcome_keeps_its_units_when_observed_values_are_zero_and_one() -> None:
    from causal.analysis import interface as api
    from causal.analysis.tests.support.interface import approve, design

    source = trial().with_columns(((pl.col("cable_system_id") % 3) == 0).cast(pl.Float64).alias("turnout_rate"))
    fixed = design("randomized", {"assignment_mechanism": "individual_randomized"},
                   outcome="turnout_rate", units="score points")
    configuration = {"method": "randomized", "treatment_column": "treatment",
                     "unit_column": "cable_system_id", "treated_value": "1", "comparator_value": "0",
                     "estimator": "difference_in_means"}
    actual = api.execute(approve(source, fixed, configuration), source)
    assert actual.primary.status == "computed"
    assert actual.primary.estimates[0].units == "score points"


@pytest.mark.parametrize("treatment_name", ["assigned", "observed", "assigned arm", "treated"])
def test_public_analysis_supports_source_names_without_internal_column_collisions(
        treatment_name: str) -> None:
    from causal.analysis import interface as api
    from causal.analysis.tests.support.interface import approve, design

    original = trial()
    source = original.rename({"treatment": treatment_name, "turnout_rate": "outcome + score",
                              "cable_system_id": "unit id",
                              "randomization_stratum": "baseline::score"})
    snapshot = source.clone()
    fixed = design("randomized", {"assignment_mechanism": "individual_randomized"},
                   outcome="outcome + score")
    configuration = {"method": "randomized", "treatment_column": treatment_name,
                     "unit_column": "unit id", "treated_value": "1", "comparator_value": "0",
                     "estimator": "ancova", "precision_covariate_column": "baseline::score"}
    actual = api.execute(approve(source, fixed, configuration,
                                 diagnostics=("baseline_balance",)), source)
    data = pd.DataFrame({name: original[name].to_list() for name in original.columns})
    data["treated"] = data["treatment"].astype(float)
    expected = feols("turnout_rate ~ treated + randomization_stratum", data=data, vcov="HC2")
    assert actual.primary.status == "computed", actual.primary.explanation
    assert actual.primary.estimates[0].estimate == pytest.approx(float(expected.coef()["treated"]), abs=1e-12)
    assert actual.primary.estimates[0].standard_error == pytest.approx(float(expected.se()["treated"]), abs=1e-12)
    assert all(row.status == "computed" for row in actual.diagnostics)
    balance = next(row for row in actual.supporting_data if row.computation_id == "baseline_balance")
    assert any("baseline::score" in item.name for item in balance.measurements)
    assert "analysis_input_" not in actual.model_dump_json()
    assert source.equals(snapshot)
