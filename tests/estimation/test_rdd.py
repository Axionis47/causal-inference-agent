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

from causal.estimation import contracts as ec
from causal.estimation import rdd
from causal.estimation.packs import EstimationPackV1, load_estimation_packs
from causal.shared.contracts import ArtifactRef

REGISTRIES = Path(__file__).resolve().parents[2] / "registries"
PACKS = load_estimation_packs(REGISTRIES / "method-pack-estimation.v1.json",
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
        "role_columns": dict(ROLES), "prepared_frame_schema_id": "estimator-input.v1",
        "row_set_hash": hashlib.sha256(b"rows").hexdigest(), "contrast_ids": (CONTRAST,),
        "required_sensitivity_ids": tuple(row.branch_id for row in PACK.sensitivity_branches),
        "figure_builder_ids": PACK.figure_builder_ids, "capacity_check": ref("capacity"),
        "seed": SEED, "numerical_tolerances": {"sensitivity_magnitude": 0.25},
        "parents": (ref("manifest"),), "versions": {"schema": "estimation-plan.v1"},
        "context_manifest": ref("manifest"), "plan_revision": 1,
        "estimator_id": PACK.estimator_id, "estimator_version": PACK.estimator_version,
        "outcome_scale": "level_difference", "multiplicity_policy_id": None,
        "primary_mask_rule_id": "selected_bandwidth", "confidence_level": PACK.confidence_level,
        "uncertainty_method": PACK.uncertainty_method,
        "finite_sample_correction": PACK.finite_sample_correction,
        "estimator_parameters": dict(PACK.parameter_defaults) | {"cutoff": CUTOFF},
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
    shifted = make_plan(estimator_parameters=dict(PACK.parameter_defaults) | {"cutoff": 2.0})
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


def test_the_figure_binning_is_fixed_before_any_result(frame: pl.DataFrame) -> None:
    seen = adapter()
    builders = rdd.figure_builders(seen.fit(frame, make_plan(), PACK).harvest)
    assert set(builders) == set(PACK.figure_builder_ids)
    binned = builders["binned_outcome_summary"](None)  # type: ignore[arg-type]
    assert 0 < len(binned) <= 2 * rdd.BINS
    assert {point.category for point in binned} == {"above", "below"}
    # Changing only the outcome cannot move a bin: the registered binning is a function of the
    # running variable and the fixed bin count alone.
    moved = frame.with_columns((pl.col("y") * 3.0 + 7.0).alias("y"))
    again = rdd.figure_builders(seen.fit(moved, make_plan(), PACK).harvest)
    shifted = again["binned_outcome_summary"](None)  # type: ignore[arg-type]
    assert [point.x_value for point in shifted] == [point.x_value for point in binned]
    assert [point.y_value for point in shifted] != [point.y_value for point in binned]
    curve = builders["fitted_curve_points"](None)  # type: ignore[arg-type]
    assert 0 < len(curve) <= rdd.CURVE + 1
    density = builders["density_continuity_summary"](None)  # type: ignore[arg-type]
    assert {point.category for point in density} >= {"cutoff", "density_left", "density_right"}
