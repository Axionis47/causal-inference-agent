# Estimation manifest, plan, mask, result, evidence, judgment-ceiling, bundle, and outcome
# payloads (PRD-004 §5, §6, §10.2, §14–§17, §19.1, §26; D-083).

from __future__ import annotations

import math
from typing import Annotated, Any, Final, Literal, Self, get_args

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from causal.shared.contracts import ArtifactRef, Identity, PayloadLocator, Sha256Hex

# The closed version keys every estimation payload pins (§6.1, §19.1).
ESTIMATION_REGISTRY_KEYS: Final = (
    "artifact_types", "method_packs", "estimation_packs", "diagnostics", "sensitivities",
    "figure_builders", "capacity", "visualization_catalog", "schema", "validators")

# §14.1 execution status and policy result; §14.2 severity; §26.2 bundle kind.
ExecutionStatus = Literal["computed", "partial", "not_computable", "failed"]
PolicyResult = Literal["acceptable", "warning", "invalidating", "descriptive"]
DiagnosticSeverity = Literal[
    "required_blocking", "invalidation_guard", "qualification_guard", "descriptive"]
EvidenceKind = Literal["diagnostic", "sensitivity", "figure_data"]
ConvergenceStatus = Literal["converged", "not_applicable", "not_converged"]
# §16.1 and §16.3, most restrictive first; an overall status is the earliest one present.
JudgmentStatus = Literal[
    "failed", "not_estimable", "not_reportable", "reportable_with_qualifications", "reportable"]
JUDGMENT_ORDER: Final[tuple[JudgmentStatus, ...]] = get_args(JudgmentStatus)
EstimationOutcomeStatus = Literal[
    "complete", "not_estimable", "invalidated", "design_conflict", "failed_observability", "failed"]

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
_NonNegInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(ge=1)]
_Ids = Annotated[tuple[Identity, ...], Field(min_length=1)]
_Level = Annotated[float, Field(gt=0.0, lt=1.0)]
ParameterValue = str | int | float | bool | None
ValueMap = dict[str, ParameterValue]
CountMap = dict[str, _NonNegInt]


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("a stored quantity must be finite; NaN and infinity never serialize")
    return value


# Every float reaching an envelope payload: NaN or infinity would break canonical replay.
Finite = Annotated[float, AfterValidator(_finite)]


class EstimationError(ValueError):
    # `code` is stable; `detail_codes` carries every family (PreparationError idiom).

    def __init__(self, message: str, code: str, detail_codes: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.code, self.detail_codes = code, detail_codes


# The most restrictive ceiling present, or `reportable` when nothing caps the claim (§16.1).
def most_restrictive(ceilings: tuple[JudgmentStatus, ...]) -> JudgmentStatus:
    return next((status for status in JUDGMENT_ORDER if status in ceilings), "reportable")


# Frozen, strict base for estimation row and fragment models.
class _Row(BaseModel):
    model_config = _MODEL_CONFIG


# Base for committed estimation payloads.
class _Payload(_Row):
    def canonical_payload(self) -> dict[str, Any]:
        # Canonical-ready dict; `content_hash(self.canonical_payload())` is replay-stable.
        return self.model_dump(mode="json")


# Ordered upstream lineage and pinned versions, on every committed estimation payload (§21).
class _Lineage(_Payload):
    parents: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    versions: dict[str, Identity]


# One content-addressed object: where it lives and what it hashes to.
class ObjectRefV1(_Row):
    object_locator: PayloadLocator
    content_hash: Sha256Hex


# The approved semantic selection both the manifest and the plan pin (§6.1, §19.1).
class _ApprovedSelection(_Payload):
    method_id: Identity
    method_pack_version: Identity
    estimand_id: Identity
    population_id: Identity
    timeframe_id: Identity
    comparator_id: Identity
    outcome_id: Identity
    unit_id: Identity
    role_columns: dict[str, Identity]
    prepared_frame_schema_id: Identity
    row_set_hash: Sha256Hex
    contrast_ids: _Ids
    required_sensitivity_ids: tuple[Identity, ...]
    figure_builder_ids: _Ids
    capacity_check: ArtifactRef
    seed: int
    numerical_tolerances: dict[str, Finite]


# The closed approved-context surface every estimation step reads from (§19.1).
class EstimationContextManifestV1(_ApprovedSelection):
    schema_version: Literal["estimation-context-manifest.v1"] = "estimation-context-manifest.v1"
    experiment_design: ArtifactRef
    runnable_frame_contract: ArtifactRef
    prepared_bundle: ArtifactRef
    estimator_input_view_id: Identity
    # The approved design facts the plan carries into `estimator_parameters` (§6.1): the RDD
    # cutoff and its assignment direction, the DiD adoption time. A pack default never supplies
    # one, and the §4 gate refuses a method whose approved role needs a fact the design omits.
    method_structure: dict[str, Identity] = {}
    contribution_mask_rule_ids: _Ids
    preprocessing_rule_ids: tuple[Identity, ...]
    uncertainty_rule_id: Identity
    required_diagnostic_ids: _Ids
    judgment_rule_ids: _Ids
    result_cardinalities: CountMap
    registry_versions: dict[str, Identity]
    recipient_map: dict[str, tuple[Identity, ...]]
    # SHA-256 of the approved upstream inputs this manifest was compiled from; the
    # manifest's own content hash lives on its envelope.
    manifest_hash: Sha256Hex

    @model_validator(mode="after")
    def _closed_registry_versions(self) -> Self:
        if set(self.registry_versions) != set(ESTIMATION_REGISTRY_KEYS):
            raise ValueError(f"registry_versions must key exactly {ESTIMATION_REGISTRY_KEYS}")
        return self


# The executable specification committed before any primary outcome is read (§6.1).
class EstimationPlanV1(_ApprovedSelection, _Lineage):
    schema_version: Literal["estimation-plan.v1"] = "estimation-plan.v1"
    context_manifest: ArtifactRef
    plan_revision: _PositiveInt
    estimator_id: Identity
    estimator_version: Identity
    outcome_scale: Identity
    multiplicity_policy_id: Identity | None
    primary_mask_rule_id: Identity
    confidence_level: _Level
    uncertainty_method: Identity
    finite_sample_correction: Identity
    # Approved fixed effects, interactions, polynomial order, kernel, or bandwidth selector.
    estimator_parameters: ValueMap
    nuisance_profile_id: Identity | None
    fold_count: _PositiveInt | None
    fold_assignment_rule_id: Identity | None
    preprocessing_recipe_ids: tuple[Identity, ...]
    required_diagnostics: dict[str, DiagnosticSeverity]
    numerical_failure_rule_ids: tuple[Identity, ...]

    @model_validator(mode="after")
    def _cross_fitting_is_all_or_nothing(self) -> Self:
        folded = (self.fold_count is not None, self.fold_assignment_rule_id is not None)
        if any(folded) and not all(folded):
            raise ValueError("a cross-fitted plan pins both fold_count and fold_assignment_rule_id")
        return self


# Which frozen rows contribute to one calculation; the bit vector is a restricted object (§6.2).
class AnalysisContributionMaskV1(_Lineage):
    schema_version: Literal["analysis-contribution-mask.v1"] = "analysis-contribution-mask.v1"
    parent_row_set_hash: Sha256Hex
    calculation_id: Identity
    outcome_id: Identity
    estimator_id: Identity
    mask_rule_id: Identity
    mask_object: ObjectRefV1
    included_counts: CountMap
    noncontributing_counts: CountMap
    reason_counts: CountMap
    builder_version: Identity

    @model_validator(mode="after")
    def _reasons_account_for_every_noncontributing_row(self) -> Self:
        if sum(self.reason_counts.values()) != self.noncontributing_counts.get("row", 0):
            raise ValueError("registered non-contribution reasons must total the excluded rows")
        return self


# One primary contrast with its required uncertainty inlined (§6.3, §26.2).
class PrimaryContrastResultV1(_Row):
    contrast_id: Identity
    estimand_id: Identity
    estimand_label: str
    estimate: Finite
    estimate_units: Identity
    comparator_id: Identity
    effect_direction: Identity
    standard_error: Annotated[float, Field(ge=0.0), AfterValidator(_finite)]
    confidence_level: _Level
    interval_lower: Finite
    interval_upper: Finite
    p_value: Annotated[float, Field(ge=0.0, le=1.0)] | None
    uncertainty_method: Identity
    finite_sample_correction: Identity
    contributing_counts: CountMap
    contribution_mask: ArtifactRef
    estimator_id: Identity
    estimator_version: Identity
    estimator_parameters: ValueMap
    adapter_version: Identity
    convergence: ConvergenceStatus
    # Fold count, effective sample size, cohort aggregation, bandwidths, kernel, and the like.
    method_quantities: ValueMap

    @model_validator(mode="after")
    def _interval_brackets_the_estimate(self) -> Self:
        if not self.interval_lower <= self.estimate <= self.interval_upper:
            raise ValueError("the confidence interval must bracket the point estimate")
        return self


# The single primary result artifact for one method and estimand family (§6.3).
class PrimaryAnalysisResultV1(_Lineage):
    schema_version: Literal["primary-analysis-result.v1"] = "primary-analysis-result.v1"
    plan: ArtifactRef
    method_id: Identity
    estimator_id: Identity
    outcome_id: Identity
    estimand_family: Identity
    primary_items: Annotated[tuple[PrimaryContrastResultV1, ...], Field(min_length=1)]
    contrast_order: _Ids
    multiplicity_result: ArtifactRef | None
    complete: bool

    @model_validator(mode="after")
    def _items_are_the_prespecified_contrasts_in_order(self) -> Self:
        if tuple(item.contrast_id for item in self.primary_items) != self.contrast_order:
            raise ValueError("primary_items must be exactly the prespecified contrasts, in order")
        return self


# The registered multiplicity adjustment over one confirmatory contrast family (§9.1).
class MultiplicityResultV1(_Lineage):
    schema_version: Literal["multiplicity-result.v1"] = "multiplicity-result.v1"
    plan: ArtifactRef
    policy_id: Identity
    # Contrast id to its adjusted p-value, adjusted bounds, and rejection decision.
    adjusted_by_contrast: dict[str, ValueMap]


# Deterministic fold assignment; the row-to-fold mapping is a restricted object (§10.2).
class CrossFitAssignmentV1(_Lineage):
    schema_version: Literal["cross-fit-assignment.v1"] = "cross-fit-assignment.v1"
    plan: ArtifactRef
    fold_count: _PositiveInt
    assignment_algorithm_id: Identity
    stratification_rule_ids: tuple[Identity, ...]
    mapping_object: ObjectRefV1
    # Fold id to its train, validation, and treated counts.
    counts_by_fold: dict[str, CountMap]
    preprocessing_recipe_version: Identity
    nuisance_profile_id: Identity


# The §14.1 result surface a diagnostic and a sensitivity branch both record.
class _EvidenceResult(_Lineage):
    plan: ArtifactRef
    primary_result: ArtifactRef
    execution_status: ExecutionStatus
    policy_result: PolicyResult
    denominators: CountMap
    contribution_mask_hash: Sha256Hex | None
    values: ValueMap
    warnings: tuple[Identity, ...]
    interpreting_rule_id: Identity
    implementation_version: Identity
    numerical_environment: ArtifactRef


# One post-estimation diagnostic result under its prespecified severity (§14).
class DiagnosticResultV1(_EvidenceResult):
    schema_version: Literal["diagnostic-result.v1"] = "diagnostic-result.v1"
    diagnostic_id: Identity
    diagnostic_version: Identity
    severity: DiagnosticSeverity
    threshold_context: ValueMap

    @model_validator(mode="after")
    def _severity_bounds_the_policy_result(self) -> Self:
        if self.policy_result == "invalidating" and self.severity != "invalidation_guard":
            raise ValueError("only an invalidation_guard diagnostic may return `invalidating`")
        return self


# One prespecified sensitivity branch and its comparison against the primary result (§15).
class SensitivityResultV1(_EvidenceResult):
    schema_version: Literal["sensitivity-result.v1"] = "sensitivity-result.v1"
    branch_id: Identity
    purpose: str
    parameter_delta: ValueMap
    # The branch's own estimate, uncertainty, estimator, and mask; absent when it did not compute.
    result: PrimaryContrastResultV1 | None
    comparison_rule_id: Identity
    comparison_result: Identity
    qualification_rule_ids: tuple[Identity, ...]

    @model_validator(mode="after")
    def _a_computed_branch_reports_its_result(self) -> Self:
        if (self.result is not None) is not (self.execution_status == "computed"):
            raise ValueError("a branch reports a result if and only if it computed")
        return self


# One typed figure-ready point; PRD-005 renders these and calculates nothing (§17).
class FigureDataPointV1(_Row):
    series_id: Identity
    category: Identity | None
    x_value: Finite | None
    y_value: Finite | None
    interval_lower: Finite | None
    interval_upper: Finite | None
    denominator: _NonNegInt | None


# The frozen data behind one required visual-evidence family (§17).
class FigureDataArtifactV1(_Lineage):
    schema_version: Literal["figure-data-artifact.v1"] = "figure-data-artifact.v1"
    visual_evidence_id: Identity
    builder_id: Identity
    builder_version: Identity
    points: Annotated[tuple[FigureDataPointV1, ...], Field(max_length=2000)]
    units: dict[str, Identity]
    labels: dict[str, str]
    # The registered aggregation, binning, and suppression rules this payload obeyed.
    rule_ids: _Ids
    contributing_counts: CountMap
    contribution_mask_hash: Sha256Hex | None
    disclosure_status: JudgmentStatus


# The one bundle shape the diagnostic, sensitivity, and figure-data fan-ins share (§26.2).
class EvidenceBundleV1(_Lineage):
    schema_version: Literal["estimation-evidence-bundle.v1"] = (
        "estimation-evidence-bundle.v1")
    kind: EvidenceKind
    plan: ArtifactRef
    results: tuple[ArtifactRef, ...]
    terminal_status_counts: CountMap

    @model_validator(mode="after")
    def _every_result_has_one_terminal_status(self) -> Self:
        counts = self.terminal_status_counts
        if set(counts) - set(get_args(ExecutionStatus)) or sum(counts.values()) != len(self.results):
            raise ValueError(f"a {self.kind} bundle counts every result under one known status")
        return self


# The deterministic ceiling for one primary contrast (§16.1).
class CeilingItemV1(_Row):
    contrast_id: Identity
    ceiling: JudgmentStatus
    triggering_rule_ids: tuple[Identity, ...]
    evidence: tuple[ArtifactRef, ...]


# Per-item and overall ceilings, fixed before any model-authored interpretation (§16.1).
class JudgmentCeilingV1(_Lineage):
    schema_version: Literal["judgment-ceiling.v1"] = "judgment-ceiling.v1"
    plan: ArtifactRef
    primary_result: ArtifactRef | None
    items: tuple[CeilingItemV1, ...]
    overall_ceiling: JudgmentStatus

    @model_validator(mode="after")
    def _overall_is_the_most_restrictive_item(self) -> Self:
        ceilings = tuple(item.ceiling for item in self.items)
        if ceilings and self.overall_ceiling != most_restrictive(ceilings):
            raise ValueError("the overall ceiling is the most restrictive required item")
        return self


# The exact runtime one estimation run executed in (§6.4).
class NumericalEnvironmentManifestV1(_Payload):
    schema_version: Literal["numerical-environment.v1"] = "numerical-environment.v1"
    python_version: Identity
    # Every pinned package version, the BLAS/LAPACK implementation included (§6.4).
    package_versions: dict[str, Identity]
    platform: Identity
    seeds: dict[str, int]
    # Estimator thread and process counts (§6.4).
    parallelism: CountMap
    float_dtype: Identity
    serialization_policy: Identity
    numerical_tolerances: dict[str, Finite]
    build_identifier: Identity
    runtime_image_digest: Identity | None


# The PRD-005 handoff: every estimation identity and hash in one record (§5.1, §26.2).
class EstimationBundleV1(_Lineage):
    schema_version: Literal["estimation-bundle.v1"] = "estimation-bundle.v1"
    experiment_design: ArtifactRef
    runnable_frame_contract: ArtifactRef
    prepared_bundle: ArtifactRef
    row_set_hash: Sha256Hex
    capacity_check: ArtifactRef
    context_manifest: ArtifactRef
    plan: ArtifactRef
    contribution_masks: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    cross_fit_assignments: tuple[ArtifactRef, ...]
    primary_result: ArtifactRef
    multiplicity_result: ArtifactRef | None
    # One EvidenceBundleV1 per kind: diagnostic, sensitivity, figure_data (§26.2).
    evidence_bundles: Annotated[tuple[ArtifactRef, ...], Field(min_length=3, max_length=3)]
    judgment_ceiling: ArtifactRef
    claim_judgment: ArtifactRef
    numerical_environment: ArtifactRef


# The estimation stage's single terminal record (§5.2).
class EstimationOutcomeV1(_Payload):
    schema_version: Literal["estimation-outcome.v1"] = "estimation-outcome.v1"
    status: EstimationOutcomeStatus
    context_manifest: ArtifactRef
    estimation_bundle: ArtifactRef | None
    design_conflict: ArtifactRef | None
    stage_run_id: Identity
    graph_thread_id: Identity
    error_code: Identity | None

    @model_validator(mode="after")
    def _status_matches_refs(self) -> Self:
        if (self.estimation_bundle is not None) is not (self.status == "complete"):
            raise ValueError("estimation_bundle is present if and only if status is complete")
        if (self.design_conflict is not None) is not (self.status == "design_conflict"):
            raise ValueError("design_conflict is present iff status is design_conflict")
        return self
