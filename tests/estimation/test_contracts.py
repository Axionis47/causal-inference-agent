# Estimation payload contracts: strict round-trips and boundary rules (T-022 §2; PRD-004 §5–§17).

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from causal.estimation import contracts as ec
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef

HASH = "a" * 64
REF = ArtifactRef(artifact_id="art-1", content_hash=HASH)
OBJECT = ec.ObjectRefV1(object_locator="objects/" + HASH, content_hash=HASH)
VERSIONS = dict.fromkeys(ec.ESTIMATION_REGISTRY_KEYS, "v1")
LINEAGE: dict[str, Any] = {"parents": (REF,), "versions": {"schema": "v1"}}

SELECTION: dict[str, Any] = {
    "method_id": "aipw", "method_pack_version": "aipw-pack.v1", "estimand_id": "ate",
    "population_id": "pop-1", "timeframe_id": "tf-1", "comparator_id": "cmp-1",
    "outcome_id": "out-1", "unit_id": "unit-1", "role_columns": {"treatment": "treat"},
    "prepared_frame_schema_id": "aipw-prepared-frame.v1", "row_set_hash": HASH,
    "contrast_ids": ("primary",), "required_sensitivity_ids": ("alternative_nuisance_profile",),
    "figure_builder_ids": ("overlap_bins",), "capacity_check": REF, "seed": 7,
    "numerical_tolerances": {"score": 1e-09},
}
MANIFEST: dict[str, Any] = {
    **SELECTION, "experiment_design": REF, "runnable_frame_contract": REF, "prepared_bundle": REF,
    "estimator_input_view_id": "aipw-estimator-input.v1", "preprocessing_rule_ids": (),
    "contribution_mask_rule_ids": ("cross_fit_predicted",), "manifest_hash": HASH,
    "uncertainty_rule_id": "influence_function_variance", "registry_versions": VERSIONS,
    "required_diagnostic_ids": ("cross_fit_score_integrity",), "judgment_rule_ids": ("ceiling.v1",),
    "result_cardinalities": {"primary_items": 1}, "recipient_map": {"claim_review": ("context",)},
}
PLAN: dict[str, Any] = {
    **SELECTION, **LINEAGE, "context_manifest": REF, "plan_revision": 1, "fold_count": 5,
    "estimator_id": "aipw_score_estimator", "estimator_version": "aipw-score-estimator.v1",
    "outcome_scale": "difference", "multiplicity_policy_id": None, "confidence_level": 0.95,
    "primary_mask_rule_id": "cross_fit_predicted", "estimator_parameters": {"estimand": "ate"},
    "uncertainty_method": "influence_function_variance", "nuisance_profile_id": "regularized_glm",
    "finite_sample_correction": "cross_fit_degrees_of_freedom", "fold_assignment_rule_id": "hash",
    "preprocessing_recipe_ids": ("estimator_scoped_recipe",),
    "required_diagnostics": {"cross_fit_score_integrity": "required_blocking"},
    "numerical_failure_rule_ids": ("solver_did_not_converge",),
}
MASK: dict[str, Any] = {
    **LINEAGE, "parent_row_set_hash": HASH, "calculation_id": "primary", "outcome_id": "out-1",
    "estimator_id": "aipw_score_estimator", "mask_rule_id": "cross_fit_predicted",
    "mask_object": OBJECT, "included_counts": {"row": 90}, "builder_version": "mask-builder.v1",
    "noncontributing_counts": {"row": 10}, "reason_counts": {"outcome_missing": 6, "support": 4},
}
CONTRAST: dict[str, Any] = {
    "contrast_id": "primary", "estimand_id": "ate", "estimand_label": "average treatment effect",
    "estimate": 1.5, "estimate_units": "outcome_units", "comparator_id": "cmp-1",
    "effect_direction": "treated_minus_comparison", "standard_error": 0.5, "p_value": 0.003,
    "confidence_level": 0.95, "interval_lower": 0.52, "interval_upper": 2.48,
    "uncertainty_method": "influence_function_variance", "contribution_mask": REF,
    "finite_sample_correction": "cross_fit_degrees_of_freedom", "convergence": "converged",
    "contributing_counts": {"row": 90}, "estimator_id": "aipw_score_estimator",
    "estimator_version": "aipw-score-estimator.v1", "estimator_parameters": {"estimand": "ate"},
    "adapter_version": "sklearn-adapter.v1", "method_quantities": {"fold_count": 5},
}
ITEM = ec.PrimaryContrastResultV1(**CONTRAST)
PRIMARY: dict[str, Any] = {
    **LINEAGE, "plan": REF, "method_id": "aipw", "estimator_id": "aipw_score_estimator",
    "outcome_id": "out-1", "estimand_family": "average_treatment_effect", "complete": True,
    "primary_items": (ITEM,), "contrast_order": ("primary",), "multiplicity_result": None,
}
EVIDENCE: dict[str, Any] = {
    **LINEAGE, "plan": REF, "primary_result": REF, "execution_status": "computed",
    "policy_result": "acceptable", "denominators": {"row": 90}, "contribution_mask_hash": HASH,
    "values": {"score_mean": 0.0}, "warnings": (), "interpreting_rule_id": "score_integrity.v1",
    "implementation_version": "diagnostics.v1", "numerical_environment": REF,
}
DIAGNOSTIC: dict[str, Any] = {
    **EVIDENCE, "diagnostic_id": "cross_fit_score_integrity", "threshold_context": {},
    "diagnostic_version": "cross-fit-score-integrity.v1", "severity": "required_blocking",
}
SENSITIVITY: dict[str, Any] = {
    **EVIDENCE, "branch_id": "alternative_nuisance_profile", "purpose": "alternative learners",
    "parameter_delta": {"nuisance_profile_id": "histogram_gradient_boosting"}, "result": ITEM,
    "comparison_rule_id": "sign_and_interval_stability", "comparison_result": "stable",
    "qualification_rule_ids": (),
}
CROSS_FIT: dict[str, Any] = {
    **LINEAGE, "plan": REF, "fold_count": 1, "assignment_algorithm_id": "stratified_hash",
    "stratification_rule_ids": ("treatment",), "mapping_object": OBJECT,
    "counts_by_fold": {"0": {"train": 72, "validation": 18}}, "nuisance_profile_id": "glm",
    "preprocessing_recipe_version": "recipe.v1",
}
FIGURE: dict[str, Any] = {
    **LINEAGE, "visual_evidence_id": "overlap", "builder_id": "overlap_bins",
    "builder_version": "overlap-bins.v1", "units": {"y_value": "count"},
    "points": (ec.FigureDataPointV1(series_id="treated", category="bin-1", x_value=0.5,
                                    y_value=12.0, interval_lower=None, interval_upper=None,
                                    denominator=90),),
    "labels": {"y_value": "units"}, "rule_ids": ("fixed_decile_bins",),
    "contributing_counts": {"row": 90}, "contribution_mask_hash": HASH,
    "disclosure_status": "reportable",
}
BUNDLE: dict[str, Any] = {
    **LINEAGE, "experiment_design": REF, "runnable_frame_contract": REF, "prepared_bundle": REF,
    "row_set_hash": HASH, "capacity_check": REF, "context_manifest": REF, "plan": REF,
    "contribution_masks": (REF,), "cross_fit_assignments": (REF,), "primary_result": REF,
    "multiplicity_result": None, "evidence_bundles": (REF, REF, REF), "judgment_ceiling": REF,
    "claim_judgment": REF, "numerical_environment": REF,
}
EVIDENCE_BUNDLE: dict[str, Any] = {
    **LINEAGE, "kind": "diagnostic", "plan": REF, "results": (REF,),
    "terminal_status_counts": {"computed": 1},
}
CEILING: dict[str, Any] = {
    **LINEAGE, "plan": REF, "primary_result": REF, "overall_ceiling": "reportable",
    "items": (ec.CeilingItemV1(contrast_id="primary", ceiling="reportable",
                               triggering_rule_ids=(), evidence=(REF,)),),
}
OUTCOME: dict[str, Any] = {
    "status": "complete", "context_manifest": REF, "estimation_bundle": REF,
    "design_conflict": None, "stage_run_id": "run-1", "graph_thread_id": "thread-1",
    "error_code": None,
}
PAYLOADS: tuple[tuple[type[BaseModel], dict[str, Any]], ...] = (
    (ec.EstimationContextManifestV1, MANIFEST), (ec.EstimationPlanV1, PLAN),
    (ec.AnalysisContributionMaskV1, MASK), (ec.PrimaryAnalysisResultV1, PRIMARY),
    (ec.MultiplicityResultV1, {**LINEAGE, "plan": REF, "policy_id": "holm",
                               "adjusted_by_contrast": {"primary": {"adjusted_p": 0.006}}}),
    (ec.CrossFitAssignmentV1, CROSS_FIT), (ec.DiagnosticResultV1, DIAGNOSTIC),
    (ec.SensitivityResultV1, SENSITIVITY), (ec.FigureDataArtifactV1, FIGURE),
    (ec.EvidenceBundleV1, EVIDENCE_BUNDLE), (ec.JudgmentCeilingV1, CEILING),
    (ec.NumericalEnvironmentManifestV1, {"python_version": "3.12.8", "float_dtype": "float64",
                                         "package_versions": {"numpy": "2.5.2"},
                                         "platform": "darwin-arm64", "seeds": {"plan": 7},
                                         "parallelism": {"threads": 1, "processes": 1},
                                         "serialization_policy": "canonical-json.v1",
                                         "numerical_tolerances": {"score": 1e-09},
                                         "build_identifier": "abc123",
                                         "runtime_image_digest": None}),
    (ec.EstimationBundleV1, BUNDLE), (ec.EstimationOutcomeV1, OUTCOME),
)


def reject(model: type[BaseModel], fields: dict[str, Any], **overrides: Any) -> str:
    with pytest.raises(ValidationError) as excinfo:
        model(**{**fields, **overrides})
    return str(excinfo.value)


def hashed(payload: Any) -> str:
    return content_hash(payload.canonical_payload())


@pytest.mark.parametrize(("model", "fields"), PAYLOADS)
class TestEveryPayload:
    def test_round_trips_and_hashes_the_same_way_twice(
        self, model: type[BaseModel], fields: dict[str, Any]
    ) -> None:
        payload = model(**fields)
        assert model.model_validate_json(payload.model_dump_json()) == payload
        assert hashed(payload) == hashed(model(**fields))

    def test_rejects_an_unknown_field(
        self, model: type[BaseModel], fields: dict[str, Any]
    ) -> None:
        assert "Extra inputs are not permitted" in reject(model, fields, invented_field="x")

    def test_is_frozen(self, model: type[BaseModel], fields: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            model(**fields).schema_version = "other.v1"  # type: ignore[attr-defined]


def test_the_manifest_pins_every_registry_version() -> None:
    assert ec.EstimationContextManifestV1(**MANIFEST).registry_versions == VERSIONS
    assert "registry_versions must key exactly" in reject(
        ec.EstimationContextManifestV1, MANIFEST, registry_versions={"schema": "v1"})


def test_the_plan_hash_changes_only_when_the_plan_changes() -> None:
    assert hashed(ec.EstimationPlanV1(**PLAN)) != hashed(
        ec.EstimationPlanV1(**{**PLAN, "plan_revision": 2}))
    assert "fold_count and fold_assignment_rule_id" in reject(
        ec.EstimationPlanV1, PLAN, fold_assignment_rule_id=None)


def test_the_mask_reasons_total_the_excluded_rows() -> None:
    assert "non-contribution reasons must total" in reject(
        ec.AnalysisContributionMaskV1, MASK, reason_counts={"outcome_missing": 6})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"interval_upper": 1.2}, "bracket the point estimate"),
        ({"estimate": float("nan")}, "must be finite"),
        ({"standard_error": -0.1}, "greater than or equal to 0"),
        ({"p_value": 1.4}, "less than or equal to 1"),
        ({"convergence": "maybe"}, "Input should be"),
    ],
)
def test_a_primary_contrast_refuses_an_unusable_quantity(
    overrides: dict[str, Any], message: str
) -> None:
    assert message in reject(ec.PrimaryContrastResultV1, CONTRAST, **overrides)


def test_the_primary_result_is_the_prespecified_contrasts_in_order() -> None:
    assert "at least 1 item" in reject(ec.PrimaryAnalysisResultV1, PRIMARY, primary_items=())
    assert "exactly the prespecified contrasts" in reject(
        ec.PrimaryAnalysisResultV1, PRIMARY, contrast_order=("secondary",))


def test_an_evidence_bundle_counts_every_result_under_one_known_kind_and_status() -> None:
    assert "Input should be" in reject(ec.EvidenceBundleV1, EVIDENCE_BUNDLE, kind="uncertainty")
    for counts in ({"computed": 2}, {"invented": 1}):
        assert "counts every result" in reject(
            ec.EvidenceBundleV1, EVIDENCE_BUNDLE, terminal_status_counts=counts)


def test_a_descriptive_diagnostic_cannot_invalidate() -> None:
    assert "invalidation_guard" in reject(
        ec.DiagnosticResultV1, DIAGNOSTIC, severity="descriptive", policy_result="invalidating")
    assert "if and only if it computed" in reject(
        ec.SensitivityResultV1, SENSITIVITY, execution_status="failed")


def test_the_overall_ceiling_is_the_most_restrictive_item() -> None:
    items = (ec.CeilingItemV1(contrast_id="a", ceiling="reportable", triggering_rule_ids=(),
                              evidence=()),
             ec.CeilingItemV1(contrast_id="b", ceiling="not_reportable",
                              triggering_rule_ids=("density_manipulation_test",), evidence=(REF,)))
    capped = ec.JudgmentCeilingV1(**{**CEILING, "items": items,
                                     "overall_ceiling": "not_reportable"})
    assert capped.overall_ceiling == "not_reportable"
    assert "most restrictive" in reject(ec.JudgmentCeilingV1, CEILING, items=items)
    assert ec.most_restrictive(()) == "reportable"
    assert ec.most_restrictive(("not_reportable", "failed")) == "failed"


def test_the_outcome_carries_the_artifact_its_status_promises() -> None:
    assert "if and only if status is complete" in reject(
        ec.EstimationOutcomeV1, OUTCOME, status="invalidated")
    assert "iff status is design_conflict" in reject(
        ec.EstimationOutcomeV1, OUTCOME, status="design_conflict", estimation_bundle=None,
        design_conflict=None)
    assert "Input should be" in reject(ec.EstimationOutcomeV1, OUTCOME, status="prepared")
