"""The complete accepted scientific snapshot survives preparation and blocks edits."""
from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from causal.analysis import interface as api
from causal.analysis.contracts import (
    AnalysisSpecification,
    ApprovedPlan,
    Assessment,
    BoundaryError,
    CandidateAssertion,
    CandidateDraft,
    CandidateOutcome,
    CompiledPlan,
    DatasetIdentity,
    FixedCandidate,
    FixedDesign,
    Outcome,
    VariableBinding,
)


@pytest.fixture()
def candidate() -> CandidateDraft:
    return CandidateDraft(
        method="randomized", population="Protocol eligible participants",
        unit_grain="One row per randomized participant", estimand="itt",
        population_policy="Retain every randomized participant",
        missingness_policy="Retain missing outcomes and report attrition",
        outcome=CandidateOutcome(column="y", kind="continuous", units="points",
                                 meaning="Prespecified follow-up score"),
        facts=(CandidateAssertion(name="assignment_mechanism", value="individual_randomized",
                                  evidence=("protocol:assignment",)),),
        configuration={"method": "randomized", "treatment_column": "arm", "unit_column": "id",
                       "treated_value": "treated", "comparator_value": "control",
                       "estimator": "difference_in_means"},
        bindings=(VariableBinding(role="treatment", column="source arm", expected_alias="arm",
                                  source_references=("notebook:arm",)),),
        seed=17, candidate_reference="candidate:r1", context_reference="notebook:r8",
        source_dataset_reference="source-snapshot:r3")


@pytest.fixture()
def data() -> pl.DataFrame:
    return pl.DataFrame({"id": list(range(24)), "arm": ["control", "treated"] * 12,
                         "y": [2.0 + 3.0 * (i % 2) + i / 10 for i in range(24)]})


def _specification(candidate: CandidateDraft, data: pl.DataFrame) -> AnalysisSpecification:
    fixed = FixedCandidate.from_candidate(candidate, reference="protocol:accepted:r1")
    result = api.assess_specification(fixed, {"dataset": api.identify_data(data, "prepared:r1")})
    assert result.status == "ready", result.issues
    assert result.specification is not None
    return result.specification


def test_projection_is_derived_and_round_trips(candidate: CandidateDraft, data: pl.DataFrame) -> None:
    specification = _specification(candidate, data)
    assert specification.schema_version == "analysis-specification.v2"
    assert specification.configuration.treatment_column == "arm"
    assert specification.seed == 17
    assert AnalysisSpecification.model_validate_json(specification.model_dump_json()) == specification
    assert specification.design.reference == "protocol:accepted:r1"


@pytest.mark.parametrize("update", [
    {"treated_value": "other"}, {"comparator_value": "other"}, {"unit_column": "household"},
    {"estimand": "ate"}, {"precision_covariate_column": "baseline"},
    {"estimator": "ancova"}, {"treatment_column": "other arm"},
])
def test_configuration_cannot_change_under_fixed_reference(
        candidate: CandidateDraft, data: pl.DataFrame, update: dict[str, Any]) -> None:
    specification = _specification(candidate, data)
    proposal = specification.model_dump()
    proposal["configuration"].update(update)
    assessment = api.assess_specification(specification.design, proposal)
    assert assessment.status == "rejected"
    assert assessment.specification is None and assessment.specification_hash is None


@pytest.mark.parametrize("update", [
    {"seed": 18}, {"diagnostics": ("baseline_balance",)},
    {"sensitivities": ("sensitivity_covariance_profile",)},
])
def test_computation_selection_cannot_change_under_fixed_reference(
        candidate: CandidateDraft, data: pl.DataFrame, update: dict[str, Any]) -> None:
    specification = _specification(candidate, data)
    assessment = api.assess_specification(specification.design, specification.model_dump() | update)
    assert assessment.status == "rejected" and assessment.specification is None


@pytest.mark.parametrize("field,value", [
    ("population", "Different participants"), ("unit_grain", "One row per household"),
    ("population_policy", "Remove participants with missing outcomes"),
    ("missingness_policy", "Impute the outcome"), ("estimand", "ate"),
    ("seed", 24), ("bindings", ()), ("facts", ()),
    ("source_dataset_reference", "another-source"),
])
def test_complete_scientific_payload_is_hashed(
        candidate: CandidateDraft, data: pl.DataFrame, field: str, value: Any) -> None:
    specification = _specification(candidate, data)
    proposal = specification.model_dump()
    proposal["design"]["candidate"][field] = value
    assessment = api.assess_specification(specification.design, proposal)
    assert assessment.status == "rejected" and assessment.specification is None
    assert any("hash" in issue.finding for issue in assessment.issues)


def test_rehashed_changed_snapshot_still_conflicts_with_supplied_fixed_design(
        candidate: CandidateDraft, data: pl.DataFrame) -> None:
    specification = _specification(candidate, data)
    changed = FixedCandidate.from_candidate(candidate.model_copy(update={"seed": 18}),
                                           reference=specification.design.reference)
    assessment = api.assess_specification(specification.design,
        {"design": changed, "dataset": specification.dataset})
    assert assessment.status == "rejected" and assessment.specification is None
    assert any(issue.field == "design" for issue in assessment.issues)


def test_freezing_detaches_original_and_nested_tampering_is_detected(
        candidate: CandidateDraft, data: pl.DataFrame) -> None:
    specification = _specification(candidate, data)
    candidate.configuration["treated_value"] = "edited later"
    assert specification.configuration.treated_value == "treated"
    assert isinstance(specification.design, FixedCandidate)
    specification.design.candidate.configuration["treated_value"] = "tampered snapshot"
    assessment = api.assess_specification(specification.design, specification)
    assert assessment.status == "rejected" and assessment.specification is None


def test_prepared_frame_binding_preserves_the_scientific_snapshot(
        candidate: CandidateDraft, data: pl.DataFrame) -> None:
    specification = _specification(candidate, data)
    changed_frame = data.with_columns(pl.lit("provenance").alias("unused"))
    rebound = api.assess_specification(specification.design,
        {"dataset": api.identify_data(changed_frame, "prepared:r2")})
    assert rebound.status == "ready" and rebound.specification is not None
    assert rebound.specification.design == specification.design
    assert rebound.specification.dataset != specification.dataset
    assert api.preflight(rebound.specification, changed_frame).ready
    assert not api.preflight(specification, changed_frame).ready


def test_expected_alias_cannot_be_repaired_by_silent_role_rebinding(
        candidate: CandidateDraft, data: pl.DataFrame) -> None:
    specification = _specification(candidate, data)
    renamed = data.rename({"arm": "source arm"})
    rebound = api.assess_specification(specification.design,
        {"dataset": api.identify_data(renamed, "prepared:r2")})
    assert rebound.specification is not None
    readiness = api.preflight(rebound.specification, renamed)
    assert not readiness.ready
    assert any(issue.category == "missing_data" and issue.field == "arm" for issue in readiness.issues)


def test_unresolved_candidate_cannot_be_fixed_or_emit_executable_acceptance(
        candidate: CandidateDraft) -> None:
    with pytest.raises(BoundaryError):
        FixedCandidate.from_candidate(CandidateDraft(method="randomized"), "protocol:incomplete")
    fixed = FixedCandidate.from_candidate(candidate, "protocol:complete")
    assessment = api.assess_specification(fixed, {})
    assert assessment.status == "needs_information"
    assert assessment.specification is None and assessment.specification_hash is None


def test_historical_readers_cannot_certify_or_execute_new_work(data: pl.DataFrame) -> None:
    old = FixedDesign(reference="historical:protocol", content_hash="a" * 64,
                      method="randomized", population="Historical population",
                      outcome=Outcome(column="y", kind="continuous", units="points"))
    specification = AnalysisSpecification.model_validate({
        "schema_version": "analysis-specification.v1", "design": old,
        "dataset": DatasetIdentity(name="historical", content_hash="b" * 64),
        "configuration": {"method": "randomized"}})
    assert AnalysisSpecification.model_validate_json(specification.model_dump_json()) == specification
    historical_response = Assessment.model_validate({
        "status": "needs_information", "capability_version": "randomized.v1",
        "specification_hash": "c" * 64, "specification": specification, "issues": ()})
    assert historical_response.schema_version == "analysis-assessment.v1"
    assert Assessment.model_validate_json(historical_response.model_dump_json()) == historical_response
    assessment = api.assess_specification(old, specification)
    assert assessment.status == "rejected" and assessment.specification is None
    plan = CompiledPlan(schema_version="analysis-plan.v2", specification=specification,
                        capability_version="randomized.v1", implementation_hash="c" * 64,
                        preflight_hash="d" * 64, diagnostics=(), sensitivities=())
    assert CompiledPlan.model_validate_json(plan.model_dump_json()) == plan
    approved = ApprovedPlan(plan=plan, approved_hash=plan.plan_hash,
                            approved_by="historical reviewer", approval_reference="historical:approval")
    with pytest.raises(BoundaryError, match="Historical"):
        api.execute(approved, data)
    with pytest.raises(ValidationError):
        AnalysisSpecification.model_validate({"design": old, "dataset": specification.dataset,
                                              "configuration": {"method": "randomized"}})


def test_current_compilation_rejects_historical_readiness_and_runs_checks_only_later(
        candidate: CandidateDraft, data: pl.DataFrame) -> None:
    specification = _specification(candidate, data)
    readiness = api.preflight(specification, data)
    plan = api.compile_plan(specification, readiness)
    assert plan.schema_version == "analysis-plan.v3"
    assert all(row.applicability_boundary == "design" and row.measurement_boundary == "execution"
               and row.evaluation_stage == "execution" for row in plan.diagnostics)
    with pytest.raises(BoundaryError):
        api.compile_plan(specification, readiness.model_copy(update={"schema_version": "analysis-preflight.v1"}))


def test_fixed_snapshot_remains_readable_after_capability_retirement(
        candidate: CandidateDraft, data: pl.DataFrame) -> None:
    retired = FixedCandidate(
        reference="protocol:retired", candidate=candidate, capability_version="randomized.retired",
        content_hash=FixedCandidate.hash_payload(candidate, "protocol:retired", "randomized.retired"))
    restored = FixedCandidate.model_validate_json(retired.model_dump_json())
    assert restored == retired
    assessment = api.assess_specification(restored, {"dataset": api.identify_data(data, "study")})
    assert assessment.status == "rejected" and assessment.specification is None
    assert any(issue.field == "design.capability_version" for issue in assessment.issues)


def test_valid_hash_and_schema_do_not_grant_design_acceptance(data: pl.DataFrame) -> None:
    candidate = CandidateDraft(method="randomized")
    version = next(row.capability_version for row in api.list_methods() if row.method == "randomized")
    incomplete = FixedCandidate(reference="protocol:unaccepted", candidate=candidate,
        capability_version=version,
        content_hash=FixedCandidate.hash_payload(candidate, "protocol:unaccepted", version))
    assessment = api.assess_specification(incomplete, {"dataset": api.identify_data(data, "study")})
    assert assessment.status == "needs_information" and assessment.specification is None


@pytest.mark.parametrize("update", [{"seed": "17"}, {"diagnostics": None}, {"sensitivities": "invalid"}])
def test_invalid_projection_values_remain_visible_as_feedback(
        candidate: CandidateDraft, data: pl.DataFrame, update: dict[str, Any]) -> None:
    specification = _specification(candidate, data)
    assessment = api.assess_specification(specification.design, specification.model_dump() | update)
    assert assessment.status == "rejected" and assessment.specification is None


def test_model_copy_cannot_bypass_compile_or_reapproved_execution(
        candidate: CandidateDraft, data: pl.DataFrame) -> None:
    specification = _specification(candidate, data)
    readiness = api.preflight(specification, data)
    plan = api.compile_plan(specification, readiness)
    altered = specification.model_copy(update={"seed": 18})
    with pytest.raises(BoundaryError):
        api.compile_plan(altered, readiness)
    altered_plan = plan.model_copy(update={"specification": altered})
    approved = ApprovedPlan(plan=altered_plan, approved_hash=altered_plan.plan_hash,
                            approved_by="reviewer", approval_reference="approval:tampered")
    with pytest.raises(BoundaryError):
        api.execute(approved, data)


def test_current_assessment_contract_never_exposes_unaccepted_specifications(
        candidate: CandidateDraft, data: pl.DataFrame) -> None:
    specification = _specification(candidate, data)
    with pytest.raises(ValidationError, match="Only ready assessment"):
        Assessment(schema_version="analysis-assessment.v2", status="rejected",
                   capability_version=None, specification=specification,
                   specification_hash=None, issues=())
    with pytest.raises(ValidationError, match="Ready assessment requires"):
        Assessment(status="ready", capability_version=None, specification=None,
                   specification_hash=None, issues=())
    with pytest.raises(ValidationError, match="Historical assessments cannot certify"):
        Assessment(schema_version="analysis-assessment.v1", status="ready",
                   capability_version=None, specification=specification,
                   specification_hash=None, issues=())
