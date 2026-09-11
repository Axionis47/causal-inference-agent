"""Complete legacy integration wall contexts and artifact builders."""

from __future__ import annotations

from typing import Any

from causal.analysis.integration import RESOURCE_ROOT
from causal.analysis.integration import contracts as ec
from causal.analysis.integration import walls as ew
from causal.analysis.integration.tests.support.plans import (
    PACK,
    ROW_HASH,
    manifest,
    plan,
    ref,
)

RULES = ew.load_validation_rules(RESOURCE_ROOT / "estimation-validation-rules.v1.json")


MANIFEST, PLAN = manifest(), plan()


OBJECT = ec.ObjectRefV1(object_locator="s3://bucket/mask", content_hash=ROW_HASH)


LINEAGE: dict[str, Any] = {"parents": (ref("plan"),), "versions": {"schema": "pinned.v1"}}


EVIDENCE: dict[str, Any] = LINEAGE | {
    "plan": ref("plan"), "primary_result": ref("result"), "denominators": {"row": 90},
    "contribution_mask_hash": None, "values": {}, "warnings": (),
    "interpreting_rule_id": "interpret.v1", "implementation_version": "impl.v1",
    "numerical_environment": ref("environment")}





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
    **LINEAGE, compiled_design=MANIFEST.compiled_design,
    prepared_bundle=MANIFEST.prepared_bundle, row_set_hash=MANIFEST.row_set_hash,
    capacity_report=MANIFEST.capacity_report, context_manifest=ref("context_manifest"),
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
        "sensitivities": tuple(sensitivity(name) for name in PLAN.required_sensitivity_ids)}
    return ew.WallContext(**(fields | over))

