# The estimation engine: contribution masks, the severity applier, the sensitivity loop, the
# §6.4 environment manifest, balance, and evidence bundles
# (T-024 §2; PRD-004 §6.2, §6.4, §14, §15, §26.2).

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from causal.analysis.common import legacy_engine as engine
from causal.analysis.integration import contracts as ec
from causal.analysis.tests.support.builders import (
    FRAME,
    MASK_OBJECT,
    PACK,
    ROLES,
    ROW_HASH,
    base,
    digest,
    make_item,
    make_plan,
    ref,
)
from causal.shared.canonical import content_hash


def mask_of(rule_id: str = "outcome_observed", *, params: ec.ValueMap | None = None,
            frame_hash: str = ROW_HASH, **over: Any) -> ec.AnalysisContributionMaskV1:
    bits = engine.mask_bits(rule_id, FRAME, ROLES, params or {})
    return engine.contribution_mask(
        make_plan(), rule_id, bits, MASK_OBJECT, frame_row_set_hash=frame_hash,
        calculation_id="primary", parents=(ref("plan"),), unit_ids=FRAME["uid"], **over)


# -- contribution masks (§6.2) --------------------------------------------


def test_the_outcome_observed_mask_accounts_for_every_frozen_row() -> None:
    mask = mask_of()
    assert (mask.included_counts["row"], mask.noncontributing_counts["row"]) == (3, 1)
    assert (mask.included_counts["unit"], mask.noncontributing_counts["unit"]) == (3, 1)
    assert mask.reason_counts == {"outcome_observed_excluded": 1}
    assert mask.parent_row_set_hash == ROW_HASH and mask.mask_rule_id == "outcome_observed"


@pytest.mark.parametrize(("rule_id", "params", "included"), [
    ("arm_membership", {}, 4), ("selected_bandwidth", {"cutoff": 1.0, "bandwidth": 0.5}, 2),
    ("donut_exclusion", {"cutoff": 1.0, "bandwidth": 1.0, "donut_radius": 0.5}, 2),
    ("group_time_cell", {"min_cell_units": 2}, 0), ("event_time_cell", {}, 4)])
def test_every_registered_mask_rule_builds_its_shape(
    rule_id: str, params: dict[str, Any], included: int
) -> None:
    assert mask_of(rule_id, params=params).included_counts["row"] == included


def test_the_mask_object_payload_is_replay_stable_and_carries_the_bits() -> None:
    # The bit vector lives only in the restricted object; the payload replays byte for byte.
    first = engine.mask_object_payload("outcome_observed",
                                       engine.mask_bits("outcome_observed", FRAME, ROLES, {}))
    second = engine.mask_object_payload("outcome_observed",
                                        engine.mask_bits("outcome_observed", FRAME, ROLES, {}))
    assert content_hash(first) == content_hash(second)
    assert first["row_count"] == 4 and isinstance(first["bits_hex"], str)


def test_a_mask_over_a_frame_outside_the_frozen_row_set_is_refused() -> None:
    with pytest.raises(ec.EstimationError) as error:
        mask_of(frame_hash=digest("another frame"))
    assert error.value.code == engine.MASK_ROW_SET_MISMATCH


def test_an_unregistered_mask_rule_fails_closed() -> None:
    with pytest.raises(ec.EstimationError) as error:
        engine.mask_bits("whatever_helps", FRAME, ROLES, {})
    assert error.value.code == engine.UNKNOWN_MASK_RULE


def test_the_estimator_view_holds_only_the_declared_role_columns() -> None:
    view = engine.estimator_view(FRAME, ROLES)
    assert "spare" not in view.columns and set(view.columns) == set(ROLES.values())
    with pytest.raises(ec.EstimationError) as error:
        engine.estimator_view(FRAME, {"outcome": "absent"})
    assert error.value.code == engine.MISSING_ROLE_COLUMN


# -- severity, diagnostics, and sensitivities (§14, §15) ------------------


@pytest.mark.parametrize(("rule_id", "branch", "expected"), [
    ("sign_and_interval_stability", make_item(1.0, 0.5, 1.5), "stable"),
    ("sign_and_interval_stability", make_item(-1.0, -1.5, -0.5), "direction_changed"),
    ("interval_width_stability", make_item(9.0, 8.0, 10.0), "intervals_disjoint"),
    ("interval_width_stability", make_item(1.2, 0.9, 1.6), "stable"),
    ("influence_stability", make_item(4.0, 3.0, 5.0), "magnitude_shifted"),
    ("null_effect_expected", make_item(0.1, -0.5, 0.5), "stable"),
    ("null_effect_expected", make_item(2.0, 1.5, 2.5), "null_effect_rejected"),
    ("descriptive_only", make_item(), "not_applicable"),
    ("sign_and_interval_stability", None, "not_computed")])
def test_the_comparison_rules_read_direction_magnitude_and_intervals(
    rule_id: str, branch: ec.PrimaryContrastResultV1 | None, expected: str
) -> None:
    assert engine.compare(rule_id, make_item(), branch, 0.25) == expected


def test_an_unregistered_comparison_rule_fails_closed() -> None:
    with pytest.raises(ec.EstimationError) as error:
        engine.compare("looks_favorable", make_item(), make_item(), 0.25)
    assert error.value.code == engine.UNKNOWN_COMPARISON_RULE


class StubAdapter:
    # A registered adapter stand-in: one item per fit, and a chosen branch that dies.

    def __init__(self, poisoned: str = "") -> None:
        self.poisoned, self.calls = poisoned, 0

    def fit(self, view: pl.DataFrame, plan: ec.EstimationPlanV1, pack: Any,
            overrides: ec.ValueMap | None = None) -> engine.AdapterResult:
        self.calls += 1
        if self.poisoned and self.poisoned in (overrides or {}):
            raise RuntimeError("the branch did not converge")
        return engine.AdapterResult(items=(make_item(1.1, 0.6, 1.6),), harvest={})


def test_every_prespecified_branch_reports_a_terminal_result_even_when_it_fails() -> None:
    # §15: a failed branch is reported failed, never omitted, and the loop never stops early.
    plan, adapter = make_plan(), StubAdapter(poisoned="specification")
    results = engine.run_sensitivities(plan, PACK, adapter, FRAME, make_item(), base())
    assert tuple(row.branch_id for row in results) == plan.required_sensitivity_ids
    reported = {row.branch_id: row for row in results}
    assert reported["unadjusted_itt"].execution_status == "failed"
    assert reported["unadjusted_itt"].result is None
    assert reported["unadjusted_itt"].warnings == (engine.BRANCH_FAILED,)
    assert reported["sensitivity_covariance_profile"].execution_status == "computed"
    assert reported["sensitivity_covariance_profile"].comparison_result == "stable"
    assert reported["approved_subgroup_contrast"].policy_result == "descriptive"
    assert adapter.calls == len(plan.required_sensitivity_ids)


# -- judgment ceiling (§16.1) ---------------------------------------------


# -- environment, balance, and bundles (§6.4, §9.3, §26.2) --


def test_the_numerical_environment_is_identical_on_two_builds() -> None:
    plan = make_plan()
    first = engine.numerical_environment(plan, ("polars", "not-installed"),
                                         build_identifier="causal-0.1.0")
    second = engine.numerical_environment(plan, ("polars", "not-installed"),
                                          build_identifier="causal-0.1.0")
    assert first.canonical_payload() == second.canonical_payload()
    assert first.package_versions["not-installed"] == "absent"
    assert first.seeds == {"plan": plan.seed} and first.parallelism == {
        "threads": 2, "processes": 1}
    assert first.numerical_tolerances == plan.numerical_tolerances


def test_the_balance_helper_reports_grouped_means_and_a_standardized_difference() -> None:
    frame = pl.DataFrame({"arm": ["a", "a", "b", "b"], "age": [10.0, 20.0, 30.0, 40.0],
                          "w": [1.0, 1.0, 1.0, 1.0]})
    plain = engine.balance(frame, "arm", ("age",))["age"]
    assert (plain["mean_low"], plain["mean_high"], plain["mean_difference"]) == (15.0, 35.0, 20.0)
    assert plain["standardized_difference"] == pytest.approx(20.0 / plain["pooled_std"])
    assert engine.balance(frame, "arm", ("age",), weight_column="w")["age"] == plain


@pytest.mark.parametrize("dtype", [pl.String, pl.Categorical, pl.Enum(["1", "900"])])
@pytest.mark.parametrize("weighted", [False, True])
def test_categorical_balance_is_level_prevalence_and_invariant_to_category_renaming(
        dtype: Any, weighted: bool) -> None:
    frame = pl.DataFrame({"arm": [0, 0, 0, 1, 1, 1], "stratum": ["1", "1", "900", "1", "900", "900"],
                          "age": [20, 25, 30, 21, 26, 31], "weight": [1.0, 2.0, 1.0, 2.0, 1.0, 1.0]})
    frame = frame.with_columns(pl.col("stratum").cast(dtype))
    weight = "weight" if weighted else None
    rows = engine.balance(frame, "arm", ("stratum", "age"), weight_column=weight)
    assert set(rows) == {"stratum::'1'", "stratum::'900'", "age"}
    renamed = frame.with_columns(pl.col("stratum").cast(pl.String).replace(
        {"1": "Zebra", "900": "Apple"}).alias("stratum"))
    again = engine.balance(renamed, "arm", ("stratum", "age"), weight_column=weight)
    assert rows["stratum::'1'"] == again["stratum::'Zebra'"]
    assert rows["stratum::'900'"] == again["stratum::'Apple'"]
    assert rows["age"] == again["age"] == engine.balance(
        frame, "arm", ("age",), weight_column=weight)["age"]
    expected_low, expected_high = (0.75, 0.5) if weighted else (2 / 3, 1 / 3)
    assert rows["stratum::'1'"]["mean_low"] == pytest.approx(expected_low)
    assert rows["stratum::'1'"]["mean_high"] == pytest.approx(expected_high)


def test_missing_category_is_a_separate_balance_indicator() -> None:
    frame = pl.DataFrame({"arm": [0, 0, 1, 1], "stratum": ["None", None, "None", "None"]})
    rows = engine.balance(frame, "arm", ("stratum",))
    assert set(rows) == {"stratum::'None'", "stratum::None"}
    assert rows["stratum::None"]["mean_low"] == 0.5
    assert rows["stratum::None"]["mean_high"] == 0.0


def test_evidence_bundle_counts_each_terminal_result_once() -> None:
    bundle = engine.evidence_bundle(
        make_plan(), "diagnostic", ((ref("d1"), "computed"), (ref("d2"), "failed"),
                                    (ref("d3"), "computed")),
        plan_ref=ref("plan"), parents=(ref("plan"),))
    assert bundle.terminal_status_counts == {"computed": 2, "failed": 1}
    assert len(bundle.results) == 3 and bundle.kind == "diagnostic"
