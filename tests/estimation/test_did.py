# The difference-in-differences adapter: simultaneous parity against a hand-computed 2x2, the
# staggered Sun-Abraham aggregate and its distinctness from an unqualified two-way fixed-effects
# coefficient, the lead and pre-period surfaces, the §11.2 refusals, and the §17 row-three
# payloads (T-028 §2; PRD-004 §11, §17).

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pyfixest.estimation import feols

from causal.estimation import contracts as ec
from causal.estimation import did
from causal.estimation.packs import load_estimation_packs
from causal.shared.contracts import ArtifactRef

REGISTRIES = Path(__file__).resolve().parents[2] / "registries"
PACKS = load_estimation_packs(REGISTRIES / "method-pack-estimation.v1.json",
                              REGISTRIES / "method-packs.v1.json")
PACK = PACKS.get("did", "did-pack.v1")
CONTRAST = "adopters_vs_never_treated"
ROLES = {"outcome": "y", "treatment": "d", "unit_identifier": "uid", "time": "period",
         "group": "grp", "adoption_time": "adopt", "cluster": "uid"}
# The known cohort effects the staggered fixture below carries, by adoption period.
EFFECTS = {4.0: 1.0, 7.0: 3.0}
SEED = 20260826


def ref(name: str) -> ArtifactRef:
    return ArtifactRef(artifact_id=name, content_hash=hashlib.sha256(name.encode()).hexdigest())


def make_plan(**over: Any) -> ec.EstimationPlanV1:
    # One frozen §6.1 DiD plan; every test varies only the parameters it is about.
    fields: dict[str, Any] = {
        "method_id": PACK.method_id, "method_pack_version": PACK.pack_version,
        "estimand_id": "att", "population_id": "stores", "timeframe_id": "wave_one",
        "comparator_id": "never_treated", "outcome_id": "sales", "unit_id": "store",
        "role_columns": dict(ROLES), "prepared_frame_schema_id": "estimator-input.v1",
        "row_set_hash": hashlib.sha256(b"rows").hexdigest(), "contrast_ids": (CONTRAST,),
        "required_sensitivity_ids": tuple(row.branch_id for row in PACK.sensitivity_branches),
        "figure_builder_ids": PACK.figure_builder_ids, "capacity_check": ref("capacity"),
        "seed": SEED, "numerical_tolerances": {"sensitivity_magnitude": 0.25},
        "parents": (ref("manifest"),), "versions": {"schema": "estimation-plan.v1"},
        "context_manifest": ref("manifest"), "plan_revision": 1,
        "estimator_id": PACK.estimator_id, "estimator_version": PACK.estimator_version,
        "outcome_scale": "level_difference", "multiplicity_policy_id": None,
        "primary_mask_rule_id": "group_time_cell", "confidence_level": PACK.confidence_level,
        "uncertainty_method": PACK.uncertainty_method,
        "finite_sample_correction": PACK.finite_sample_correction,
        "estimator_parameters": dict(PACK.parameter_defaults), "nuisance_profile_id": None,
        "fold_count": None, "fold_assignment_rule_id": None, "preprocessing_recipe_ids": (),
        "required_diagnostics": PACK.severities(),
        "numerical_failure_rule_ids": PACK.not_estimable_rule_ids}
    return ec.EstimationPlanV1(**(fields | over))


def staggered_plan(**params: Any) -> ec.EstimationPlanV1:
    return make_plan(estimator_parameters=dict(PACK.parameter_defaults) | {
        "adoption_profile_id": "staggered"} | params)


def rows(records: list[tuple[Any, ...]]) -> pl.DataFrame:
    return pl.DataFrame(records, schema=["uid", "period", "grp", "adopt", "d", "y"], orient="row")


@pytest.fixture(scope="module")
def two_by_two() -> pl.DataFrame:
    # Two periods, two groups, one common adoption at period 2. Every value is a function of the
    # row, so the 2x2 difference below is pinned by the fixture and not by a seed.
    found = []
    for unit in range(24):
        treated = unit % 2
        for period in (1.0, 2.0):
            outcome = 5.0 + 0.25 * unit + 1.3 * period + (2.5 if treated and period == 2 else 0.0)
            found.append((f"u{unit:03d}", period, float(treated), 2.0 * treated,
                          1.0 if (treated and period == 2) else 0.0, outcome))
    return rows(found)


@pytest.fixture(scope="module")
def staggered() -> pl.DataFrame:
    # Three cohorts with heterogeneous effects: adopters at period 4 gain 1.0, adopters at
    # period 7 gain 3.0, and a third of the units are never treated.
    rng = np.random.default_rng(SEED)
    found = []
    for unit in range(72):
        cohort = [4.0, 7.0, 0.0][unit % 3]
        level = rng.normal(0.0, 0.4)
        for period in range(1, 11):
            treated = bool(cohort) and period >= cohort
            outcome = (level + 0.1 * period + (EFFECTS[cohort] if treated else 0.0)
                       + rng.normal(0.0, 0.05))
            found.append((f"u{unit:03d}", float(period), cohort, cohort,
                          1.0 if treated else 0.0, outcome))
    return rows(found)


def adapter() -> did.DifferenceInDifferencesAdapter:
    return did.DifferenceInDifferencesAdapter(ref("mask"))


def number(values: ec.ValueMap, key: str) -> float:
    found = values[key]
    assert isinstance(found, int | float)
    return float(found)


def hand_computed(frame: pl.DataFrame) -> float:
    # The 2x2 difference in differences, computed from cell means without any estimator.
    means = {(group, period): value for group, period, value in frame.group_by("grp", "period").agg(
        pl.col("y").mean().alias("mean")).sort("grp", "period").iter_rows()}
    return ((means[(1.0, 2.0)] - means[(1.0, 1.0)])
            - (means[(0.0, 2.0)] - means[(0.0, 1.0)]))


def twfe(frame: pl.DataFrame) -> float:
    # The unqualified two-way fixed-effects treatment coefficient PRD-004 §11.1 refuses.
    data = pd.DataFrame({name: frame[name].to_numpy() for name in ("uid", "period", "d", "y")})
    return float(feols("y ~ d | uid + period", data=data, vcov={"CRV1": "uid"}).coef()["d"])


def test_simultaneous_matches_the_hand_computed_two_by_two(two_by_two: pl.DataFrame) -> None:
    found = adapter().fit(two_by_two, make_plan(), PACK)
    item = found.items[0]
    assert item.contrast_id == CONTRAST
    assert item.estimate == pytest.approx(hand_computed(two_by_two), abs=1e-8)
    assert item.interval_lower <= item.estimate <= item.interval_upper
    assert item.method_quantities["adoption_profile_id"] == "simultaneous"
    assert item.adapter_version == did.ADAPTER_VERSION
    assert item.contributing_counts == {"row": two_by_two.height, "unit": 24}


def test_staggered_recovers_the_known_cohort_weighted_effect(staggered: pl.DataFrame) -> None:
    found = adapter().fit(staggered, staggered_plan(), PACK)
    item = found.items[0]
    # Both cohorts contribute four post-adoption event times inside the approved window, so the
    # registered cohort weighting targets the plain average of the two known effects.
    assert item.estimate == pytest.approx(2.0, abs=0.05)
    assert item.interval_lower <= item.estimate <= item.interval_upper
    assert item.method_quantities["aggregation"] == "cohort_weighted_overall"
    assert number(item.method_quantities, "aggregated_cells") == 8


def test_the_unqualified_twfe_coefficient_is_not_the_committed_primary(
        staggered: pl.DataFrame) -> None:
    item = adapter().fit(staggered, staggered_plan(), PACK).items[0]
    unqualified = twfe(staggered)
    assert abs(unqualified - item.estimate) > 0.05
    assert abs(item.estimate - 2.0) < abs(unqualified - 2.0)


def test_lead_estimates_and_the_joint_pre_period_test_surface(staggered: pl.DataFrame) -> None:
    harvest = adapter().fit(staggered, staggered_plan(), PACK).harvest
    leads = harvest["event_study_pre_period_test"]
    assert number(leads, "leads") >= 4
    assert [key for key in leads if key.startswith("lead_")]
    assert 0.0 <= number(leads, "joint_pre_period_p_value") <= 1.0
    assert number(leads, "largest_lead_estimate") < 0.1
    placement = harvest["pre_and_post_period_placement"]
    assert number(placement, "pre_periods") >= 2
    assert number(placement, "post_periods") >= 1
    assert placement["reference_period"] == did.REFERENCE


def test_a_pre_trend_is_reported_and_never_removed(staggered: pl.DataFrame) -> None:
    # A cohort-specific pre-trend must show up in the leads and in the joint test; shortening the
    # window afterwards is not an option the adapter offers.
    tilted = staggered.with_columns(
        pl.when(pl.col("adopt") == 7.0).then(pl.col("y") + 0.35 * (pl.col("period") - 7.0)
                                             .clip(upper_bound=0.0)).otherwise(pl.col("y")))
    leads = adapter().fit(tilted, staggered_plan(), PACK).harvest["event_study_pre_period_test"]
    assert number(leads, "largest_lead_estimate") > 0.2
    assert number(leads, "joint_pre_period_p_value") < 0.05


def test_an_unsupported_reference_cell_refuses_instead_of_moving_the_comparison() -> None:
    # The only adopting cohort turns on in the first period, so the approved reference period has
    # no support. The estimator refuses; it does not fall back to another comparison.
    found = []
    for unit in range(24):
        cohort = 1.0 if unit % 2 else 0.0
        for period in range(1, 5):
            found.append((f"u{unit:03d}", float(period), cohort, cohort,
                          1.0 if cohort else 0.0, 1.0 + 0.5 * period + 0.2 * unit))
    with pytest.raises(ec.EstimationError) as raised:
        adapter().fit(rows(found), staggered_plan(), PACK)
    assert raised.value.code == did.NO_PRE


def test_a_repeated_cross_section_refuses_the_panel_profile(staggered: pl.DataFrame) -> None:
    crossed = staggered.with_columns(
        pl.concat_str(pl.col("uid"), pl.col("period").cast(pl.String)).alias("uid"))
    for plan in (staggered_plan(), make_plan()):
        with pytest.raises(ec.EstimationError) as raised:
            adapter().fit(crossed, plan, PACK)
        # Both registered profiles absorb unit fixed effects, so neither pretends the units of a
        # repeated cross section persist across periods.
        assert raised.value.code == did.NOT_PANEL


def test_a_missing_never_treated_cohort_refuses_the_approved_comparison(
        staggered: pl.DataFrame) -> None:
    treated_only = staggered.filter(pl.col("adopt") > 0.0)
    with pytest.raises(ec.EstimationError) as raised:
        adapter().fit(treated_only, staggered_plan(), PACK)
    assert raised.value.code == did.NO_COMPARISON


def test_every_prespecified_branch_runs_over_the_frozen_frame(staggered: pl.DataFrame) -> None:
    plan, seen = staggered_plan(), adapter()
    primary = seen.fit(staggered, plan, PACK).items[0]
    for branch in PACK.sensitivity_branches:
        item = seen.fit(staggered, plan, PACK, dict(branch.parameter_delta)).items[0]
        assert math.isfinite(item.estimate)
        assert item.estimator_parameters | dict(branch.parameter_delta) == item.estimator_parameters
        if branch.branch_id == "placebo_period_check":
            assert abs(item.estimate) < 0.1
        if branch.branch_id == "balanced_panel_contribution":
            assert item.estimate == pytest.approx(primary.estimate, abs=0.05)


def test_the_diagnostic_harvest_answers_every_required_row(staggered: pl.DataFrame) -> None:
    harvest = adapter().fit(staggered, staggered_plan(), PACK).harvest
    assert set(PACK.severities()) <= set(harvest)
    assert harvest["unit_period_schema_reconciliation"]["structure"] == "panel"
    assert number(harvest["group_time_cohort_support"], "cohorts") == 2
    assert number(harvest["cluster_covariance_adequacy"], "clusters") == 72
    assert number(harvest["attrition_composition_change"], "composition_change_share") == 0.0
    assert number(harvest["aggregate_weight_sensitivity"], "largest_cell_weight") > 0.0
    assert harvest["reference_period_numerical_integrity"]["converged"] is True


def test_the_figure_payloads_wrap_frozen_values(staggered: pl.DataFrame) -> None:
    found = adapter().fit(staggered, staggered_plan(), PACK)
    builders = did.figure_builders(found.harvest)
    assert set(builders) == set(PACK.figure_builder_ids)
    means = builders["group_time_means"](None)  # type: ignore[arg-type]
    assert len(means) == 30
    assert {point.category for point in means} == {"0.0", "4.0", "7.0"}
    events = builders["event_time_estimates"](None)  # type: ignore[arg-type]
    assert events and all(
        point.interval_lower is not None and point.y_value is not None
        and point.interval_lower <= point.y_value <= (point.interval_upper or 0.0)
        for point in events)
    counts = builders["support_composition_counts"](None)  # type: ignore[arg-type]
    assert counts and all(point.denominator for point in counts)
    assert did.VISUAL_EVIDENCE["event_time_estimates"] == "event_time_evidence"
