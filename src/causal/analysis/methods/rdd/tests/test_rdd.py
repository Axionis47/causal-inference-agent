# The sharp-RDD adapter: parity with a direct `rdrobust` call on a deterministic fixture, the
# registered `rddensity` diagnostic, cutoff immutability under a contradicting assignment, the
# approved bandwidth branches, and the fixed §17 row-four binning (T-028 §2; PRD-004 §12, §17).

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from rddensity import rddensity
from rdrobust import rdrobust

from causal.analysis.common import legacy_diagnostics as diagnostic_policy
from causal.analysis.common import legacy_engine as engine
from causal.analysis.integration import RESOURCE_ROOT
from causal.analysis.integration import contracts as ec
from causal.analysis.integration.packs import EstimationPackV1, load_estimation_packs
from causal.analysis.methods.rdd import estimation as rdd
from causal.shared.contracts import ArtifactRef

REGISTRIES = Path(__file__).resolve().parents[6] / "registries"
PACKS = load_estimation_packs(RESOURCE_ROOT / "method-pack-estimation.v1.json",
                              REGISTRIES / "method-packs.v1.json")
PACK = PACKS.get("sharp_rdd", "sharp-rdd-pack.v1")
CONTRAST = "above_vs_below_cutoff"
ROLES = {"outcome": "y", "treatment": "d", "unit_identifier": "uid", "running_variable": "score",
         "predetermined_covariate": "x", "cluster": "site"}
# The fixture below is generated once from this seed and carries this jump at the cutoff.
SEED, CUTOFF, JUMP, ROWS = 20260826, 0.0, 1.5, 4000


def ref(name: str) -> ArtifactRef:
    return ArtifactRef(artifact_id=name, content_hash=hashlib.sha256(name.encode()).hexdigest())


def make_plan(**over: Any) -> ec.EstimationPlanV1:
    # One frozen §6.1 sharp-RDD plan; every test varies only the parameters it is about.
    fields: dict[str, Any] = {
        "method_id": PACK.method_id, "method_pack_version": PACK.pack_version,
        "estimand_id": "late", "population_id": "applicants", "timeframe_id": "wave_one",
        "comparator_id": "below_cutoff", "outcome_id": "award", "unit_id": "applicant",
        "role_columns": dict(ROLES),
        "row_set_hash": hashlib.sha256(b"rows").hexdigest(), "contrast_ids": (CONTRAST,),
        "required_sensitivity_ids": tuple(row.branch_id for row in PACK.sensitivity_branches),
        "figure_builder_ids": PACK.figure_builder_ids, "capacity_report": ref("capacity"),
        "numerical_tolerances": {"sensitivity_magnitude": 0.25},
        "parents": (ref("manifest"),), "versions": {"schema": "estimation-plan.v1"},
        "context_manifest": ref("manifest"), "plan_revision": 1, "seed": SEED,
        "estimator_id": PACK.estimator_id, "estimator_version": PACK.estimator_version,
        "outcome_scale": "level_difference", "multiplicity_policy_id": None,
        "primary_mask_rule_id": "selected_bandwidth", "confidence_level": PACK.confidence_level,
        "uncertainty_method": PACK.uncertainty_method,
        "finite_sample_correction": PACK.finite_sample_correction,
        "estimator_parameters": dict(PACK.parameter_defaults) | {
            "cutoff": CUTOFF, "assignment_direction": "above",
            "treated_value": "1.0", "comparator_value": "0.0"},
        "nuisance_profile_id": None, "fold_count": None, "fold_assignment_rule_id": None,
        "preprocessing_recipe_ids": (), "required_diagnostics": PACK.severities(),
        "numerical_failure_rule_ids": PACK.not_estimable_rule_ids}
    return ec.EstimationPlanV1(**(fields | over))


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    # One deterministic sharp design: a running variable over [-5, 5], a linear trend, and the
    # known jump above the approved cutoff.
    rng = np.random.default_rng(SEED)
    score = rng.uniform(-5.0, 5.0, ROWS)
    return pl.DataFrame({
        "uid": [f"u{index:04d}" for index in range(ROWS)], "score": score,
        "y": 2.0 + 0.4 * score + JUMP * (score >= CUTOFF) + rng.normal(0.0, 0.4, ROWS),
        "d": (score >= CUTOFF).astype(float), "x": 0.2 * score + rng.normal(0.0, 1.0, ROWS),
        "site": [f"s{index % 40}" for index in range(ROWS)]})


def adapter() -> rdd.SharpRegressionDiscontinuityAdapter:
    return rdd.SharpRegressionDiscontinuityAdapter(ref("mask"))


def number(values: ec.ValueMap, key: str) -> float:
    found = values[key]
    assert isinstance(found, int | float)
    return float(found)


def direct(frame: pl.DataFrame, **over: Any) -> Any:
    # The same call the adapter makes, issued straight against the pinned library.
    return rdrobust(y=frame["y"].to_numpy(), x=frame["score"].to_numpy(), c=CUTOFF, p=1,
                    kernel="tri", bwselect="mserd", masspoints="check", level=95,
                    cluster=frame["site"].cast(pl.Categorical).to_physical().to_numpy(), **over)


def test_the_primary_item_matches_a_direct_rdrobust_call(frame: pl.DataFrame) -> None:
    item = adapter().fit(frame, make_plan(), PACK).items[0]
    found = direct(frame)
    assert item.estimate == pytest.approx(float(found.coef.loc["Robust"].iloc[0]), rel=1e-9)
    assert item.standard_error == pytest.approx(float(found.se.loc["Robust"].iloc[0]), rel=1e-9)
    lower, upper = (float(bound) for bound in found.ci.loc["Robust"].to_list())
    assert (item.interval_lower, item.interval_upper) == pytest.approx((lower, upper), rel=1e-9)
    # The robust bias-corrected quantity is the pack's primary, and it recovers the known jump.
    assert item.method_quantities["primary_quantity"] == "robust_bias_corrected"
    assert item.estimate == pytest.approx(JUMP, abs=0.2)
    assert item.interval_lower <= JUMP <= item.interval_upper
    assert item.contrast_id == CONTRAST
    assert item.adapter_version == rdd.ADAPTER_VERSION


def test_the_selected_bandwidth_comes_from_the_approved_rule(frame: pl.DataFrame) -> None:
    item = adapter().fit(frame, make_plan(), PACK).items[0]
    found = direct(frame)
    left, right = (float(value) for value in found.bws.loc["h"].to_list())
    assert number(item.method_quantities, "bandwidth_left") == pytest.approx(left, rel=1e-9)
    assert number(item.method_quantities, "bandwidth_right") == pytest.approx(right, rel=1e-9)
    assert item.method_quantities["bandwidth_selector"] == "mserd"
    assert number(item.method_quantities, "effective_left") < ROWS


def test_the_density_test_maps_to_the_registered_diagnostic(frame: pl.DataFrame) -> None:
    harvest = adapter().fit(frame, make_plan(), PACK).harvest
    seen = rddensity(X=frame["score"].to_numpy(), c=CUTOFF)
    assert number(harvest["density_manipulation_test"], "density_p_value") == pytest.approx(
        float(seen.test["p_jk"]), rel=1e-9)
    assert number(harvest["density_manipulation_test"], "density_p_value") > 0.05
    qualification = harvest["sorting_and_other_policy_qualification"]
    assert number(qualification, "cutoff") == CUTOFF
    assert number(qualification, "density_left") > 0.0


def test_every_required_diagnostic_row_is_harvested(frame: pl.DataFrame) -> None:
    harvest = adapter().fit(frame, make_plan(), PACK).harvest
    assert set(PACK.severities()) <= set(harvest)
    assert number(harvest["cutoff_side_support"], "effective_observations_per_side") > 20
    assert number(harvest["mass_points_and_heaping"], "repeated_value_share") == 0.0
    assert number(harvest["covariate_continuity"], "covariate_jump_z") < 2.0
    assert number(harvest["bandwidth_polynomial_sensitivity"], "relative_estimate_change") < 0.5
    assert number(harvest["influence_leverage_near_cutoff"],
                  "single_observation_leverage_share") < 0.25
    integrity = harvest["robust_bias_correction_integrity"]
    assert number(integrity, "confidence_level") == 0.95
    assert number(integrity, "conventional_estimate") != number(integrity,
                                                                "bias_corrected_estimate")


def test_an_assignment_contradiction_invalidates_and_removes_no_row(
        frame: pl.DataFrame) -> None:
    tampered = frame.with_columns(pl.lit(1.0).alias("d"))
    with pytest.raises(ec.EstimationError) as raised:
        adapter().fit(tampered, make_plan(), PACK)
    assert raised.value.code == rdd.CONTRADICTION
    assert tampered.height == ROWS


def test_the_plan_cutoff_is_the_only_cutoff(frame: pl.DataFrame) -> None:
    # Moving the approved cutoff moves the estimand, and the item says which cutoff it answered.
    shifted = make_plan(estimator_parameters=dict(PACK.parameter_defaults) | {
        "cutoff": 2.0, "assignment_direction": "above",
        "treated_value": "1.0", "comparator_value": "0.0"})
    away = adapter().fit(frame.with_columns(
        (pl.col("score") >= 2.0).cast(pl.Float64).alias("d")), shifted, PACK).items[0]
    assert number(away.method_quantities, "cutoff") == 2.0
    assert abs(away.estimate) < JUMP / 2.0
    at = adapter().fit(frame, make_plan(), PACK).items[0]
    assert number(at.method_quantities, "cutoff") == CUTOFF


def test_registered_bandwidth_branches_scale_the_approved_bandwidth(
        frame: pl.DataFrame) -> None:
    plan, seen = make_plan(), adapter()
    primary = seen.fit(frame, plan, PACK).items[0]
    base = number(primary.method_quantities, "bandwidth_left")
    for branch_id, multiple in (("half_bandwidth", 0.5), ("double_bandwidth", 2.0)):
        item = seen.fit(frame, plan, PACK, dict(PACK.branch(branch_id).parameter_delta)).items[0]
        assert number(item.method_quantities, "bandwidth_left") == pytest.approx(base * multiple)
        assert item.estimate == pytest.approx(primary.estimate, abs=0.5)


def test_the_other_approved_branches_run_and_stay_labelled(frame: pl.DataFrame) -> None:
    plan, seen = make_plan(), adapter()
    for branch_id in ("local_quadratic", "alternative_kernel", "symmetric_donut",
                      "covariate_adjusted", "placebo_cutoff"):
        branch = PACK.branch(branch_id)
        item = seen.fit(frame, plan, PACK, dict(branch.parameter_delta)).items[0]
        assert item.interval_lower <= item.estimate <= item.interval_upper
        assert item.estimator_parameters | dict(branch.parameter_delta) == item.estimator_parameters
        if branch_id == "local_quadratic":
            assert number(item.method_quantities, "polynomial_order") == 2
        if branch_id == "alternative_kernel":
            assert str(item.method_quantities["kernel"]).lower().startswith("uni")
        if branch_id == "placebo_cutoff":
            assert number(item.method_quantities, "cutoff") == CUTOFF - 1.0
            assert item.interval_lower <= 0.0 <= item.interval_upper


def test_an_unregistered_branch_id_is_refused(frame: pl.DataFrame) -> None:
    with pytest.raises(ec.EstimationError) as raised:
        PACK.branch("quarter_bandwidth")
    assert raised.value.code == "unregistered_pack_reference"
    assert isinstance(PACK, EstimationPackV1)


def test_scientific_support_bins_are_fixed_independently_of_outcome(frame: pl.DataFrame) -> None:
    seen = adapter()
    harvest = seen.fit(frame, make_plan(), PACK).harvest
    binned = harvest["figure_binned_outcome"]
    cells = [key.split("|") for key in binned]
    assert 0 < len(cells) <= 2 * rdd.BINS
    assert {row[0] for row in cells} == {"above", "below"}
    assert sum(int(row[2]) for row in cells) == frame.height
    assert all(float(row[3]) <= float(row[4]) for row in cells)
    moved = frame.with_columns((pl.col("y") * 3.0 + 7.0).alias("y"))
    shifted = seen.fit(moved, make_plan(), PACK).harvest["figure_binned_outcome"]
    # Outcome changes may alter means/intervals, but never the prespecified bin positions/counts.
    locations = lambda values: sorted(tuple(key.split("|")[:3]) for key in values)
    assert locations(shifted) == locations(binned)
    assert list(shifted.values()) != list(binned.values())
    curve = harvest["figure_fitted_curve"]
    assert 0 < len(curve) <= rdd.CURVE + 1
    assert all(key.count("|") == 4 for key in curve)
    assert harvest["figure_density_continuity"]["cutoff"] == CUTOFF


def test_density_support_preserves_the_estimated_limits_and_bandwidths(frame: pl.DataFrame) -> None:
    found = adapter().fit(frame, make_plan(), PACK)
    measured = found.harvest["sorting_and_other_policy_qualification"]
    support = found.harvest["figure_density_continuity"]
    for name in ("density_left", "density_right", "density_difference"):
        assert support[name] == measured[name]
    assert support["cutoff"] == CUTOFF
    assert support["bandwidth_left"] == float(found.fit.bws.loc["h"].iloc[0])
    assert support["bandwidth_right"] == float(found.fit.bws.loc["h"].iloc[1])


def test_absent_covariates_are_unavailable_instead_of_false_balance_or_adjustment(
        frame: pl.DataFrame) -> None:
    from causal.analysis.tests.support.builders import base

    plan = make_plan(role_columns={key: value for key, value in ROLES.items()
                                  if key != "predetermined_covariate"},
                     required_sensitivity_ids=("covariate_adjusted",))
    seen = adapter().fit(frame, plan, PACK)
    assert "covariate_continuity" not in seen.harvest
    diagnostics = diagnostic_policy.run_diagnostics(PACK, seen.harvest, base())
    continuity = next(row for row in diagnostics if row.diagnostic_id == "covariate_continuity")
    assert continuity.execution_status == "not_computable"
    assert continuity.policy_result == "warning" and continuity.values == {}
    assert diagnostic_policy.qualified_inapplicable_diagnostic(plan, continuity)
    branches = engine.run_sensitivities(plan, PACK, adapter(), frame, seen.items[0], base())
    adjusted = branches[0]
    assert adjusted.execution_status == "failed" and adjusted.result is None
    assert rdd.NO_APPROVED_COVARIATES in adjusted.warnings
    assert adjusted.comparison_result == "not_computed" and adjusted.policy_result == "warning"



def test_public_sharp_rdd_preserves_the_cutoff_and_clustered_robust_estimate(
        frame: pl.DataFrame) -> None:
    from causal.analysis import interface as api
    from causal.analysis.tests.support.interface import approve, design

    fixed = design("rdd", {"sharp_assignment": True, "cutoff": CUTOFF,
                           "assignment_direction": "above"})
    configuration = {"method": "rdd", "treatment_column": "d", "unit_column": "uid",
                     "treated_value": "1.0", "comparator_value": "0.0",
                     "running_column": "score", "cutoff": CUTOFF, "running_units": "score units",
                     "cluster_column": "site"}
    actual = api.execute(approve(frame, fixed, configuration), frame)
    expected = direct(frame)
    assert actual.primary.status == "computed", actual.primary.explanation
    assert actual.primary.estimates[0].estimate == pytest.approx(
        float(expected.coef.loc["Robust"].iloc[0]), rel=1e-9)
    assert actual.primary.estimates[0].standard_error == pytest.approx(
        float(expected.se.loc["Robust"].iloc[0]), rel=1e-9)

    population = {v.name: v.value for v in actual.primary.estimates[0].population}
    assert population["input_row"] == frame.height
    assert population["local_fit_rows_left"] == int(expected.N_h[0])
    assert population["local_fit_rows_right"] == int(expected.N_h[1])
    quantities = {v.name: v.value for v in actual.primary.estimates[0].method_quantities}
    assert quantities["bandwidth_left"] == pytest.approx(float(expected.bws.loc["h"].iloc[0]))


def test_public_sensitivities_match_direct_bandwidth_kernel_polynomial_and_covariate_calls(
        frame: pl.DataFrame) -> None:
    from causal.analysis import interface as api
    from causal.analysis.tests.support.interface import approve, design

    fixed = design("rdd", {"sharp_assignment": True, "cutoff": CUTOFF, "assignment_direction": "above"})
    configuration = {"method": "rdd", "treatment_column": "d", "unit_column": "uid",
                     "treated_value": "1.0", "comparator_value": "0.0", "running_column": "score",
                     "cutoff": CUTOFF, "running_units": "score units", "cluster_column": "site",
                     "covariate_column": "x"}
    primary = direct(frame)
    widths = primary.bws.loc["h"].to_numpy()
    options = {"half_bandwidth": {"h": (widths * 0.5).tolist()},
               "double_bandwidth": {"h": (widths * 2.0).tolist()},
               "local_quadratic": {"p": 2}, "alternative_kernel": {"kernel": "uni"},
               "covariate_adjusted": {"covs": frame["x"].to_numpy().reshape(-1, 1)}}
    actual = api.execute(approve(frame, fixed, configuration, sensitivities=tuple(options)), frame)
    assert [branch.computation_id for branch in actual.sensitivities] == list(options)
    base = {"y": frame["y"].to_numpy(), "x": frame["score"].to_numpy(), "c": CUTOFF,
            "p": 1, "kernel": "tri", "bwselect": "mserd", "masspoints": "check", "level": 95,
            "cluster": frame["site"].cast(pl.Categorical).to_physical().to_numpy()}
    for branch in actual.sensitivities:
        expected = rdrobust(**(base | options[branch.computation_id]))
        assert branch.status == "computed", branch.explanation
        assert branch.estimates[0].estimate == pytest.approx(float(expected.coef.loc["Robust"].iloc[0]), rel=1e-9)
        assert branch.estimates[0].standard_error == pytest.approx(float(expected.se.loc["Robust"].iloc[0]), rel=1e-9)
