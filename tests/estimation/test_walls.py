# The fifteen §18 estimation walls over their declarative rows (T-023 §2; PRD-004 §18, §26.2).

from __future__ import annotations

from typing import Any

import pytest

from causal.estimation import contracts as ec
from causal.estimation import plancompile as pc
from causal.estimation import walls as ew
from causal.shared.registry import RegistryError
from causal.shared.validation import ValidationReport
from tests.estimation.test_plancompile import (
    CARDINALITIES,
    PACK,
    REGISTRIES,
    ROW_HASH,
    manifest,
    plan,
    ref,
    structure,
)

RULES = ew.load_validation_rules(REGISTRIES / "estimation-validation-rules.v1.json")
MANIFEST, PLAN = manifest(), plan()
OBJECT = ec.ObjectRefV1(object_locator="s3://bucket/mask", content_hash=ROW_HASH)
LINEAGE: dict[str, Any] = {"parents": (ref("plan"),), "versions": {"schema": "pinned.v1"}}
EVIDENCE: dict[str, Any] = LINEAGE | {
    "plan": ref("plan"), "primary_result": ref("result"), "denominators": {"row": 90},
    "contribution_mask_hash": None, "values": {}, "warnings": (),
    "interpreting_rule_id": "interpret.v1", "implementation_version": "impl.v1",
    "numerical_environment": ref("environment")}
CONFLICT = pc.recheck_capacity(PLAN, structure(
    cardinalities=dict(CARDINALITIES) | {"contrasts": 400}))
# The same approved context with its frozen row set moved: walls 1 and 15 must both see it.
DRIFTED = MANIFEST.model_copy(update={"row_set_hash": ROW_HASH[::-1]})


def mask(rule_id: str, parent: str = ROW_HASH) -> ec.AnalysisContributionMaskV1:
    return ec.AnalysisContributionMaskV1(
        **LINEAGE, parent_row_set_hash=parent, calculation_id="primary", outcome_id="completion",
        estimator_id=PACK.estimator_id, mask_rule_id=rule_id, mask_object=OBJECT,
        included_counts={"row": 90}, noncontributing_counts={"row": 10},
        reason_counts={"outcome_missing": 10}, builder_version="mask-builder.v1")


def item(contrast_id: str = "arm_b_vs_control", level: float = 0.95) -> ec.PrimaryContrastResultV1:
    return ec.PrimaryContrastResultV1(
        contrast_id=contrast_id, estimand_id="att", estimand_label="ATT", estimate=0.2,
        estimate_units="proportion", comparator_id="control", effect_direction="increase",
        standard_error=0.05, confidence_level=level, interval_lower=0.1, interval_upper=0.3,
        p_value=0.01, uncertainty_method=PACK.uncertainty_method,
        finite_sample_correction=PACK.finite_sample_correction, contributing_counts={"row": 90},
        contribution_mask=ref("mask"), estimator_id=PACK.estimator_id,
        estimator_version=PACK.estimator_version, estimator_parameters={},
        adapter_version="adapter.v1", convergence="converged", method_quantities={})


def result(items: tuple[ec.PrimaryContrastResultV1, ...] = (),
           complete: bool = True) -> ec.PrimaryAnalysisResultV1:
    rows = items or (item(),)
    return ec.PrimaryAnalysisResultV1(
        **LINEAGE, plan=ref("plan"), method_id=PACK.method_id, estimator_id=PACK.estimator_id,
        outcome_id="completion", estimand_family="att", primary_items=rows,
        contrast_order=tuple(row.contrast_id for row in rows), multiplicity_result=None,
        complete=complete)


def diagnostic(diagnostic_id: str, status: str = "computed") -> ec.DiagnosticResultV1:
    return ec.DiagnosticResultV1(
        **EVIDENCE, execution_status=status, policy_result="acceptable",
        diagnostic_id=diagnostic_id, diagnostic_version="diagnostic.v1",
        severity=PACK.severities()[diagnostic_id], threshold_context={})


def sensitivity(branch_id: str, status: str = "computed") -> ec.SensitivityResultV1:
    return ec.SensitivityResultV1(
        **EVIDENCE, execution_status=status, policy_result="acceptable", branch_id=branch_id,
        purpose="prespecified branch", parameter_delta={"specification": "alternative"},
        result=item() if status == "computed" else None, comparison_rule_id="compare.v1",
        comparison_result="stable", qualification_rule_ids=())


def figure(builder_id: str) -> ec.FigureDataArtifactV1:
    return ec.FigureDataArtifactV1(
        **LINEAGE, visual_evidence_id=builder_id, builder_id=builder_id,
        builder_version="builder.v1", points=(), units={}, labels={}, rule_ids=("aggregate.v1",),
        contributing_counts={"row": 90}, contribution_mask_hash=None,
        disclosure_status="reportable")


def assignment(counts: dict[str, dict[str, int]]) -> ec.CrossFitAssignmentV1:
    return ec.CrossFitAssignmentV1(
        **LINEAGE, plan=ref("plan"), fold_count=len(counts), assignment_algorithm_id="folds.v1",
        stratification_rule_ids=(), mapping_object=OBJECT, counts_by_fold=counts,
        preprocessing_recipe_version="recipe.v1", nuisance_profile_id="regularized_glm")


CEILING = ec.JudgmentCeilingV1(
    **LINEAGE, plan=ref("plan"), primary_result=ref("result"),
    items=(ec.CeilingItemV1(contrast_id="arm_b_vs_control", ceiling="reportable",
                            triggering_rule_ids=(), evidence=()),),
    overall_ceiling="reportable")
BUNDLE = ec.EstimationBundleV1(
    **LINEAGE, experiment_design=MANIFEST.experiment_design,
    runnable_frame_contract=MANIFEST.runnable_frame_contract,
    prepared_bundle=MANIFEST.prepared_bundle, row_set_hash=MANIFEST.row_set_hash,
    capacity_check=MANIFEST.capacity_check, context_manifest=ref("context_manifest"),
    plan=ref("plan"), contribution_masks=(ref("mask"),), cross_fit_assignments=(),
    primary_result=ref("result"), multiplicity_result=None,
    evidence_bundles=(ref("diagnostics"), ref("sensitivities"), ref("figures")),
    judgment_ceiling=ref("ceiling"), claim_judgment=ref("claim"),
    numerical_environment=ref("environment"))


def context(**over: Any) -> ew.WallContext:
    fields: dict[str, Any] = {
        "rules": RULES, "handoff_accepted": True, "manifest": MANIFEST, "plan": PLAN, "pack": PACK,
        "estimator_input_types": {"treatment": "categorical", "outcome": "numeric"},
        "masks": (mask(PLAN.primary_mask_rule_id),), "mask_refs": (ref("mask"),),
        "frozen_row_count": 100, "primary_result": result(),
        "numerical_environment": ref("environment"),
        "diagnostics": tuple(diagnostic(name) for name in PLAN.required_diagnostics),
        "sensitivities": tuple(sensitivity(name) for name in PLAN.required_sensitivity_ids),
        "ceiling": CEILING, "figures": tuple(figure(name) for name in PLAN.figure_builder_ids),
        "bundle": BUNDLE, "claim_validator": lambda ctx: ()}
    return ew.WallContext(**(fields | over))


# One characteristic failure per wall; wall 13's refusal has its own test below.
FAILURES: tuple[tuple[int, dict[str, Any], str], ...] = (
    (1, {"handoff_accepted": False}, "handoff_not_accepted"),
    (1, {"entry_codes": ("preparation_not_prepared",)}, "entry_validation_failed"),
    (1, {"manifest": DRIFTED}, "handoff_hash_mismatch"),
    (2, {"plan": None}, "plan_not_committed"),
    (2, {"pack": PACK.model_copy(update={"estimator_id": "swapped"})}, "plan_not_committed"),
    (2, {"plan": PLAN.model_copy(update={"primary_mask_rule_id": "invented"})},
     "plan_reference_unregistered"),
    (3, {"capacity_conflict": CONFLICT}, "delivery_capacity_conflict"),
    (4, {"estimator_input_types": {"treatment": "numeric"}}, "estimator_input_unsatisfied"),
    (4, {"structure_codes": ("cluster_mapping_unresolved",)}, "method_structure_unsatisfied"),
    (5, {"masks": (mask("never_registered"),)}, "mask_unregistered"),
    (5, {"masks": (mask(PLAN.primary_mask_rule_id, parent=ROW_HASH[::-1]),)}, "mask_unregistered"),
    (5, {"frozen_row_count": 4000}, "mask_unregistered"),
    (7, {"primary_result": result(complete=False)}, "primary_result_incomplete"),
    (7, {"primary_result": result((item("unplanned"),))}, "contrast_order_mismatch"),
    (7, {"primary_result": None}, "primary_result_incomplete"),
    (8, {"primary_result": result((item(level=0.9),))}, "uncertainty_incomplete"),
    (9, {"diagnostics": ()}, "diagnostic_not_terminal"),
    (10, {"sensitivities": ()}, "sensitivity_not_terminal"),
    (11, {"mask_refs": ()}, "result_lineage_unresolved"),
    (11, {"numerical_environment": None}, "result_lineage_unresolved"),
    (12, {"ceiling": None}, "judgment_ceiling_incomplete"),
    (14, {"figures": ()}, "figure_data_unresolved"),
    (15, {"bundle": None}, "estimation_bundle_incomplete"),
    (15, {"manifest": DRIFTED}, "estimation_bundle_incomplete"),
    (15, {"trace_codes": ("EV-P4-010",)}, "estimation_bundle_incomplete"))


def test_the_registry_covers_every_wall_and_the_report_admits_fifteen() -> None:
    assert {rule.wall for rule in RULES} == set(range(1, ew.MAX_WALL + 1))
    assert {rule.kind for rule in RULES} == set(ew.RULE_KINDS)
    assert ValidationReport(wall=ew.MAX_WALL, issues=()).wall == 15


def test_every_wall_passes_on_a_complete_green_context() -> None:
    report = ew.validate(ew.MAX_WALL, context())
    assert report.passed and report.wall == ew.MAX_WALL
    assert all(ew.wall(number, context()).passed for number in range(1, ew.MAX_WALL + 1))


@pytest.mark.parametrize(("number", "over", "code"), FAILURES)
def test_each_wall_reports_its_characteristic_failure(
        number: int, over: dict[str, Any], code: str) -> None:
    report = ew.wall(number, context(**over))
    assert not report.passed and code in {issue.code for issue in report.issues}
    assert all(issue.rule_id.startswith("est.") for issue in report.issues)


def test_wall_six_catches_a_poisoned_fold() -> None:
    folds = {"fold_1": {"train": 80, "validation": 20}, "fold_2": {"train": 80, "validation": 20}}
    clean = context(assignments=(assignment(folds),),
                    fold_fit_counts={name: {"train": 80, "validation": 0} for name in folds})
    assert ew.wall(6, clean).passed
    leaked = context(assignments=(assignment(folds),),
                     fold_fit_counts={"fold_1": {"train": 80, "validation": 0},
                                      "fold_2": {"train": 80, "validation": 3}})
    report = ew.wall(6, leaked)
    assert not report.passed and report.issues[0].code == "fold_leakage_detected"
    assert report.issues[0].artifact_ids == ("fold_2:validation_rows_fitted",)
    missing = ew.wall(6, context(assignments=(assignment(folds),)))
    assert missing.issues[0].artifact_ids == ("fold_1:fit_receipt_missing",
                                             "fold_2:fit_receipt_missing")


def test_wall_thirteen_refuses_until_the_claim_validator_lands() -> None:
    report = ew.wall(13, context(claim_validator=ew._no_claim_validator))
    assert not report.passed and report.issues[0].code == "claim_exceeds_ceiling"
    assert report.issues[0].artifact_ids == (ew.CLAIM_VALIDATOR_MISSING,)
    assert ew.wall(13, context(claim_validator=lambda ctx: ())).passed


def test_an_earlier_failure_stops_every_later_wall() -> None:
    report = ew.validate(ew.MAX_WALL, context(handoff_accepted=False, bundle=None, ceiling=None))
    assert report.wall == 1 and {issue.code for issue in report.issues} == {"handoff_not_accepted"}


def test_an_unknown_check_or_wall_fails_closed() -> None:
    rogue = RULES[0].model_copy(update={"params": {"check": "invented", "path": "/"}})
    with pytest.raises(RegistryError) as unknown_check:
        ew.wall(1, context(rules=(rogue,)))
    assert unknown_check.value.code == ew.UNKNOWN_RULE_CHECK
    with pytest.raises(RegistryError) as unknown_wall:
        ew.wall(16, context())
    assert unknown_wall.value.code == ew.UNKNOWN_WALL
