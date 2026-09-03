# The estimation engine: contribution masks, the severity applier, the sensitivity loop, the
# §16.1 ceiling, the §6.4 environment manifest, balance, and the frozen figure boundary
# (T-024 §2; PRD-004 §6.2, §6.4, §14, §15, §16.1, §17).

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from causal.estimation import contracts as ec
from causal.estimation import engine
from causal.estimation.packs import load_estimation_packs
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef

REGISTRIES = Path(__file__).resolve().parents[2] / "registries"
PACKS = load_estimation_packs(REGISTRIES / "method-pack-estimation.v1.json",
                              REGISTRIES / "method-packs.v1.json")
PACK = PACKS.get("randomized_experiment", "randomized-experiment-pack.v1")
CONTRAST = "arm_b_vs_control"
ROLES = {"outcome": "y", "treatment": "arm", "unit_identifier": "uid",
         "running_variable": "score", "group": "arm", "time": "period"}
FRAME = pl.DataFrame({"y": [1.0, None, 3.0, 4.0], "arm": ["a", "b", "a", "b"],
                      "uid": [1, 2, 3, 4], "score": [0.1, 0.9, 1.4, 2.0],
                      "period": [1, 1, 2, 2], "spare": ["x", "x", "x", "x"]})


def digest(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def ref(name: str) -> ArtifactRef:
    return ArtifactRef(artifact_id=name, content_hash=digest(name))


ROW_HASH = digest("rows")
MASK_OBJECT = ec.ObjectRefV1(object_locator=f"objects/{digest('bits')}",
                             content_hash=digest("bits"))


def make_plan(**over: Any) -> ec.EstimationPlanV1:
    # One frozen §6.1 plan; every test varies only the field it is about.
    fields: dict[str, Any] = {
        "method_id": PACK.method_id, "method_pack_version": PACK.pack_version,
        "estimand_id": "att", "population_id": "enrolled", "timeframe_id": "wave_one",
        "comparator_id": "control", "outcome_id": "completion", "unit_id": "participant",
        "role_columns": dict(ROLES), "prepared_frame_schema_id": "estimator-input.v1",
        "row_set_hash": ROW_HASH, "contrast_ids": (CONTRAST,),
        "required_sensitivity_ids": tuple(r.branch_id for r in PACK.sensitivity_branches),
        "figure_builder_ids": PACK.figure_builder_ids, "capacity_check": ref("capacity"),
        "seed": 7, "numerical_tolerances": {"sensitivity_magnitude": 0.25},
        "parents": (ref("manifest"),), "versions": {"schema": "estimation-plan.v1"},
        "context_manifest": ref("manifest"), "plan_revision": 1,
        "estimator_id": PACK.estimator_id, "estimator_version": PACK.estimator_version,
        "outcome_scale": "difference", "multiplicity_policy_id": None,
        "primary_mask_rule_id": "outcome_observed", "confidence_level": PACK.confidence_level,
        "uncertainty_method": PACK.uncertainty_method,
        "finite_sample_correction": PACK.finite_sample_correction,
        "estimator_parameters": {"threads": 2, "processes": 1}, "nuisance_profile_id": None,
        "fold_count": None, "fold_assignment_rule_id": None, "preprocessing_recipe_ids": (),
        "required_diagnostics": PACK.severities(),
        "numerical_failure_rule_ids": PACK.not_estimable_rule_ids}
    return ec.EstimationPlanV1(**(fields | over))


def make_item(estimate: float = 1.0, lower: float = 0.5, upper: float = 1.5, *,
              contrast: str = CONTRAST, convergence: str = "converged",
              ) -> ec.PrimaryContrastResultV1:
    return ec.PrimaryContrastResultV1(
        contrast_id=contrast, estimand_id="att", estimand_label="ATT", estimate=estimate,
        estimate_units="probability", comparator_id="control",
        effect_direction="higher_is_treated", standard_error=0.25, confidence_level=0.95,
        interval_lower=lower, interval_upper=upper, p_value=0.04,
        uncertainty_method=PACK.uncertainty_method, contributing_counts={"row": 100},
        finite_sample_correction=PACK.finite_sample_correction, contribution_mask=ref("mask"),
        estimator_id=PACK.estimator_id, estimator_version=PACK.estimator_version,
        estimator_parameters={}, adapter_version="rct.v1", convergence=convergence,
        method_quantities={})


def make_result(*items: ec.PrimaryContrastResultV1) -> ec.PrimaryAnalysisResultV1:
    rows = items or (make_item(),)
    return ec.PrimaryAnalysisResultV1(
        parents=(ref("plan"),), versions={"schema": "primary-analysis-result.v1"},
        plan=ref("plan"), method_id=PACK.method_id, estimator_id=PACK.estimator_id,
        outcome_id="completion", estimand_family="att", primary_items=rows,
        contrast_order=tuple(row.contrast_id for row in rows), multiplicity_result=None,
        complete=True)


def base(**over: Any) -> dict[str, Any]:
    return {"parents": (ref("plan"),), "versions": {"schema": "diagnostic-result.v1"},
            "plan": ref("plan"), "primary_result": ref("primary"),
            "denominators": {"row": 100}, "contribution_mask_hash": digest("mask"),
            "numerical_environment": ref("env")} | over


def make_diagnostic(diagnostic_id: str, severity: str, *, status: str = "computed",
                    policy: str = "acceptable") -> ec.DiagnosticResultV1:
    return ec.DiagnosticResultV1(
        **base(), diagnostic_id=diagnostic_id, diagnostic_version="v1", severity=severity,
        threshold_context={}, execution_status=status, policy_result=policy, values={},
        warnings=(), interpreting_rule_id=diagnostic_id, implementation_version="v1")


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


THRESHOLDS: ec.ValueMap = {"max_overall_attrition": 0.2, "minimum_clusters": 8}


@pytest.mark.parametrize(("severity", "values", "expected"), [
    ("descriptive", {"overall_attrition": 0.9}, "descriptive"),
    ("required_blocking", {"overall_attrition": 0.1, "clusters": 9}, "acceptable"),
    ("required_blocking", {"overall_attrition": 0.9}, "warning"),
    ("qualification_guard", {"clusters": 2}, "warning"),
    ("invalidation_guard", {"clusters": 2}, "invalidating"),
    ("invalidation_guard", {"clusters": 9}, "acceptable")])
def test_the_severity_applier_covers_every_registered_severity(
    severity: str, values: ec.ValueMap, expected: str
) -> None:
    # §14.2: severity is fixed before execution and no value may reinterpret it.
    assert engine.severity_result(severity, values, THRESHOLDS) == expected


def test_every_required_diagnostic_reaches_a_visible_terminal_result() -> None:
    # A harvested row computes, an empty harvest fails, an absent row is not computable —
    # and every registered row appears exactly once, in the pack's order.
    harvest = {"baseline_balance": {"standardized_difference": 0.02},
               "model_convergence_integrity": {}}
    results = engine.run_diagnostics(PACK, harvest, base())
    assert tuple(row.diagnostic_id for row in results) == tuple(
        row.diagnostic_id for row in PACK.required_diagnostics)
    reported = {row.diagnostic_id: row for row in results}
    assert reported["baseline_balance"].execution_status == "computed"
    assert reported["model_convergence_integrity"].execution_status == "failed"
    assert reported["covariance_cluster_adequacy"].execution_status == "not_computable"
    assert reported["covariance_cluster_adequacy"].warnings == (engine.DIAGNOSTIC_NOT_COMPUTED,)


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


def ceiling(diagnostics: tuple[ec.DiagnosticResultV1, ...], *, required: dict[str, str],
            result: ec.PrimaryAnalysisResultV1 | None = None,
            contrasts: tuple[str, ...] = (CONTRAST,)) -> ec.JudgmentCeilingV1:
    return engine.judgment_ceiling(
        make_plan(required_diagnostics=required, contrast_ids=contrasts),
        make_result() if result is None else result, diagnostics,
        {"guard": ref("guard")}, plan_ref=ref("plan"), primary_ref=ref("primary"),
        parents=(ref("plan"),))


@pytest.mark.parametrize(("diagnostics", "required", "expected"), [
    ((make_diagnostic("guard", "invalidation_guard"),),
     {"guard": "invalidation_guard"}, "reportable"),
    ((make_diagnostic("guard", "invalidation_guard", policy="invalidating"),),
     {"guard": "invalidation_guard"}, "not_reportable"),
    ((), {"guard": "required_blocking"}, "not_reportable"),
    ((make_diagnostic("guard", "qualification_guard", policy="warning"),),
     {"guard": "qualification_guard"}, "reportable_with_qualifications"),
    ((make_diagnostic("guard", "descriptive", policy="descriptive"),),
     {"guard": "descriptive"}, "reportable")])
def test_the_ceiling_truth_table_caps_each_condition(
    diagnostics: tuple[ec.DiagnosticResultV1, ...], required: dict[str, str], expected: str
) -> None:
    # §16.1: each row of the table, its rule id recorded and its evidence bound.
    found = ceiling(diagnostics, required=required)
    assert found.overall_ceiling == expected and len(found.items) == 1
    assert found.items[0].ceiling == expected
    if expected != "reportable":
        assert found.items[0].triggering_rule_ids and found.items[0].evidence


def test_an_unavailable_estimator_caps_its_own_item_at_not_estimable() -> None:
    other = make_item(contrast="arm_c_vs_control", convergence="not_converged")
    found = ceiling((make_diagnostic("guard", "descriptive", policy="descriptive"),),
                    required={"guard": "descriptive"},
                    result=make_result(make_item(), other),
                    contrasts=(CONTRAST, "arm_c_vs_control"))
    assert [item.ceiling for item in found.items] == ["reportable", "not_estimable"]
    # The overall ceiling is the most restrictive required item, never an average.
    assert found.overall_ceiling == "not_estimable"
    assert found.items[1].triggering_rule_ids == (
        f"{engine.CEILING_RULES[0]}:arm_c_vs_control",)


def test_a_missing_primary_result_leaves_every_item_not_estimable() -> None:
    found = engine.judgment_ceiling(
        make_plan(), None, (), {}, plan_ref=ref("plan"), primary_ref=None,
        parents=(ref("plan"),))
    assert found.overall_ceiling == "not_estimable"


# -- environment, balance, figures, and bundles (§6.4, §9.3, §17, §26.2) --


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


def points(found: ec.PrimaryAnalysisResultV1) -> tuple[ec.FigureDataPointV1, ...]:
    return tuple(ec.FigureDataPointV1(
        series_id=item.contrast_id, category=None, x_value=None, y_value=item.estimate,
        interval_lower=item.interval_lower, interval_upper=item.interval_upper, denominator=100)
        for item in found.primary_items)


def test_a_figure_dataset_wraps_frozen_values_and_a_bundle_counts_them() -> None:
    figure = engine.figure_data(
        make_plan(), "group_time_means", points, make_result(),
        visual_evidence_id="arm_summary", disclosure="reportable_with_qualifications",
        parents=(ref("plan"),), counts={"row": 100}, mask_hash=digest("mask"))
    assert figure.points[0].y_value == 1.0 and figure.builder_version == engine.BUILDER_VERSION and figure.units == {"x": "period", "y": "difference"} and figure.labels["x"] == "Time period"
    bundle = engine.evidence_bundle(
        make_plan(), "diagnostic", ((ref("d1"), "computed"), (ref("d2"), "failed"),
                                    (ref("d3"), "computed")),
        plan_ref=ref("plan"), parents=(ref("plan"),))
    assert bundle.terminal_status_counts == {"computed": 2, "failed": 1}
    assert len(bundle.results) == 3 and bundle.kind == "diagnostic"
