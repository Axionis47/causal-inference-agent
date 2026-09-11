"""Typed contracts for the compiler-owned design path (T-036)."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causal.design.contracts import DiagnosticResultV1, SourceInterpretationV1, TableGrain
from causal.design.semantics import RoleName
from causal.shared.contracts import (
    ArtifactRef,
    Identity,
    ReferenceKind,
    Sha256Hex,
    reference_field,
)
from causal.shared.envelope import CausalFrameV1, EpistemicStatus, EvidenceClass

_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
_Ids = Annotated[tuple[Identity, ...], Field(min_length=1)]
Scalar = str | int | float | bool


class _Row(BaseModel):
    model_config = _CONFIG


class _Payload(_Row):
    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ResolutionCategory(StrEnum):
    MODEL_FIX = "model_fix"
    HUMAN_INPUT = "human_input"
    NEEDS_DATA = "needs_data"
    UNSUPPORTED = "unsupported"
    SYSTEM_FAILURE = "system_failure"


class ResponsibleActor(StrEnum):
    MODEL = "model"
    USER = "user"
    DATA_OWNER = "data_owner"
    PRODUCT = "product"
    SYSTEM = "system"


class ValidationIssueV2(_Row):
    code: Identity
    category: ResolutionCategory
    json_path: str
    rule_id: Identity
    safe_actual_summary: str
    expected_constraint: str
    why_blocking: str
    responsible_actor: ResponsibleActor
    allowed_actions: _Ids
    candidate_values: tuple[str, ...] = ()
    required_input_ids: tuple[Identity, ...] = ()
    fingerprint: Sha256Hex

    @classmethod
    def build(
        cls, *, code: str, category: ResolutionCategory, path: str, rule_id: str,
        actual: str, expected: str, why: str, actor: ResponsibleActor,
        actions: tuple[str, ...], candidates: tuple[str, ...] = (),
        required: tuple[str, ...] = (),
    ) -> ValidationIssueV2:
        body = {"code": code, "category": category.value, "path": path, "rule": rule_id, "actual": actual, "expected": expected, "required": sorted(required)}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        return cls(code=code, category=category, json_path=path, rule_id=rule_id, safe_actual_summary=actual,
                   expected_constraint=expected, why_blocking=why, responsible_actor=actor,
                   allowed_actions=actions, candidate_values=candidates, required_input_ids=required, fingerprint=digest)


class EvidenceRelation(StrEnum):
    DIRECT = "direct"
    CORROBORATING = "corroborating"
    CONFLICTING = "conflicting"
    SPECULATIVE = "speculative"


class FactSource(StrEnum):
    MEASUREMENT = "measurement"
    USER = "user"
    DOCUMENT = "document"
    INTERPRETATION = "interpretation"
    MODEL = "model"
    DERIVED = "derived"


class FactAcceptanceStatus(StrEnum):
    ACCEPTED = "accepted"
    UNACCEPTED = "unaccepted"


class DesignFactV2(_Row):
    fact_id: Identity
    requirement_id: Annotated[Identity, reference_field(ReferenceKind.REQUIREMENT)]
    scope_id: Identity
    value: Scalar | tuple[str, ...] | None
    source: FactSource
    source_artifact_ids: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.ARTIFACT)
    ]
    supporting_evidence_ids: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.EVIDENCE)
    ] = ()
    evidence_class: EvidenceClass | None
    relation: EvidenceRelation
    epistemic_status: EpistemicStatus
    acceptance_status: FactAcceptanceStatus
    executable: bool

    @model_validator(mode="after")
    def _executable_has_non_model_support(self) -> Self:
        accepted = self.acceptance_status is FactAcceptanceStatus.ACCEPTED
        if self.executable is not accepted:
            raise ValueError("a fact is executable exactly when it is accepted")
        if self.executable and (self.value is None or self.source is FactSource.MODEL or self.relation is EvidenceRelation.CONFLICTING):
            raise ValueError("an executable fact needs a value and non-model, non-conflicting support")
        return self


class RoleBindingV2(_Row):
    role: RoleName
    columns: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.COLUMN, min_length=1)
    ]
    concept_id: Annotated[Identity, reference_field(ReferenceKind.CONCEPT)]
    source_artifact_ids: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.ARTIFACT, min_length=1)
    ]
    epistemic_status: EpistemicStatus
    derived: bool = False


class DesignFactSetV2(_Payload):
    schema_version: Literal["design-fact-set.v2"] = "design-fact-set.v2"
    selected_csv: ArtifactRef
    grain: TableGrain
    facts: tuple[DesignFactV2, ...]
    role_bindings: tuple[RoleBindingV2, ...]
    conflicts: tuple[Identity, ...]

    @model_validator(mode="after")
    def _unique_authoritative_values(self) -> Self:
        executable = [fact.fact_id for fact in self.facts if fact.executable]
        if len(executable) != len(set(executable)):
            raise ValueError("an executable fact id must have exactly one value")
        roles = [row.role for row in self.role_bindings]
        if len(roles) != len(set(roles)):
            raise ValueError("a causal role must have exactly one compiled binding")
        return self

    def fact(self, fact_id: str) -> Scalar | tuple[str, ...] | None:
        return next((row.value for row in self.facts if row.fact_id == fact_id
                     and row.executable), None)

    def columns(self, role: RoleName | str) -> tuple[str, ...]:
        name = RoleName(role)
        return next((row.columns for row in self.role_bindings if row.role is name), ())


class DiagnosticAssessmentV2(_Row):
    diagnostic_result_id: Annotated[
        Identity, reference_field(ReferenceKind.DIAGNOSTIC_RESULT)
    ]
    judgment: Literal["supports", "cautions", "rules_out", "not_decisive"]


class DiagnosticObservationV2(_Row):
    diagnostic_result_id: Annotated[
        Identity, reference_field(ReferenceKind.DIAGNOSTIC_RESULT)
    ]
    result: DiagnosticResultV1


class DiagnosticObservationSetV2(_Payload):
    schema_version: Literal["diagnostic-observation-set.v2"] = "diagnostic-observation-set.v2"
    selected_csv: ArtifactRef
    observations: Annotated[tuple[DiagnosticObservationV2, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _unique_result_ids(self) -> Self:
        ids = [row.diagnostic_result_id for row in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("a diagnostic result id occurs exactly once")
        return self


class AgentDesignProposalV2(_Payload):
    schema_version: Literal["agent-design-proposal.v2"] = "agent-design-proposal.v2"
    assignment_mechanism: Literal["randomized", "self_selected", "policy_cutoff",
                                  "time_of_adoption", "unknown"]
    requested_estimand: Literal["itt", "ate", "att",
                                "att_group_time_aggregate", "late_at_cutoff", "unknown"]
    comparator: str
    ranked_method_ids: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.METHOD, min_length=1)
    ]
    method_facts: tuple[AgentFactProposalV2, ...]
    optional_assumption_ids: tuple[Identity, ...]
    optional_risk_ids: tuple[Identity, ...]
    optional_sensitivity_ids: tuple[Identity, ...]
    requested_diagnostic_ids: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.DIAGNOSTIC, max_length=4)
    ] = ()
    diagnostic_assessments: tuple[DiagnosticAssessmentV2, ...] = ()
    source_interpretations: tuple[SourceInterpretationV1, ...] = ()

    @model_validator(mode="after")
    def _interpretations_match_proposed_facts(self) -> Self:
        proposed: dict[str, Scalar | None] = {
            "assignment_mechanism": self.assignment_mechanism,
            "estimand": self.requested_estimand,
            "comparator": self.comparator,
            **{row.fact_id: row.value for row in self.method_facts},
        }
        keys = [(row.fact_key, row.evidence_id) for row in self.source_interpretations]
        if len(keys) != len(set(keys)):
            raise ValueError("a source may be interpreted once per fact")
        for row in self.source_interpretations:
            if row.fact_key not in proposed or row.value != proposed[row.fact_key]:
                raise ValueError("every source interpretation must match one proposed fact")
            if (row.relation in {"direct", "corroborating"}
                    and proposed[row.fact_key] in {None, "unknown"}):
                raise ValueError("unknown facts cannot have supporting interpretations")
        return self


class AgentFactProposalV2(_Row):
    fact_id: Literal["cutoff", "sharp_assignment", "adoption_time"]
    value: Scalar | None


class BoundDiagnosticInputV2(_Row):
    parameter: Identity
    source_kind: Literal["role", "fact", "literal"]
    source_id: Identity
    columns: Annotated[tuple[Identity, ...], reference_field(ReferenceKind.COLUMN)] = ()
    value: Scalar | None = None


class DiagnosticPlanItemV2(_Row):
    diagnostic_id: Annotated[Identity, reference_field(ReferenceKind.DIAGNOSTIC)]
    primitive: Identity
    required_for_eligibility: bool
    inputs: tuple[BoundDiagnosticInputV2, ...]


class DiagnosticPlanV2(_Payload):
    schema_version: Literal["diagnostic-plan.v2"] = "diagnostic-plan.v2"
    selected_csv: ArtifactRef
    candidate_method_id: Annotated[Identity, reference_field(ReferenceKind.METHOD)]
    items: Annotated[tuple[DiagnosticPlanItemV2, ...], Field(min_length=1)]
    issues: tuple[ValidationIssueV2, ...]


class DiagnosticReportV2(_Payload):
    schema_version: Literal["diagnostic-report.v2"] = "diagnostic-report.v2"
    selected_csv: ArtifactRef
    candidate_method_id: Annotated[Identity, reference_field(ReferenceKind.METHOD)]
    results: tuple[DiagnosticResultV1, ...]
    issues: tuple[ValidationIssueV2, ...]
    computable: bool


class PreparationPolicyV2(_Row):
    output_grain: TableGrain
    key_columns: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.COLUMN, min_length=1)
    ]
    required_roles: tuple[RoleName, ...]
    protected_columns: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.COLUMN, min_length=1)
    ]
    imputation_permitted: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.COLUMN)
    ]
    eligibility_rule_ids: tuple[Identity, ...]
    unusable_row_rule_ids: tuple[Identity, ...]
    required_missingness_indicators: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.COLUMN)
    ]
    method_structure: dict[str, Scalar]
    deletion_impact_dimensions: _Ids
    required_final_diagnostic_ids: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.DIAGNOSTIC, min_length=1)
    ]
    estimator_input_schema_id: Identity


class ColumnMeasurementV2(_Row):
    label: str
    units: str | None
    scale: str | None
    source_card: ArtifactRef
    supporting_evidence_ids: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.EVIDENCE)
    ]


class CompiledDesignV2(_Payload):
    schema_version: Literal["compiled-design.v2"] = "compiled-design.v2"
    selected_csv: ArtifactRef
    causal_question: str
    intended_decision: str
    frame: CausalFrameV1
    method_id: Annotated[Identity, reference_field(ReferenceKind.METHOD)]
    method_pack_version: Identity
    estimand: Identity
    comparator: str
    unit: str
    primary_contrasts: _Ids
    role_bindings: tuple[RoleBindingV2, ...]
    column_measurements: dict[str, ColumnMeasurementV2] = {}
    preparation: PreparationPolicyV2
    assumptions: tuple[str, ...]
    identification_risks: tuple[str, ...]
    sensitivity_requirements: tuple[str, ...]
    rejected_methods: dict[str, str]
    # Historical designs retain their original values; new designs leave presentation open.
    required_visual_evidence: tuple[Identity, ...] = ()
    multiplicity_policy_id: Identity | None
    registry_versions: dict[str, Identity]


class CapacityValueV2(_Row):
    dimension: Identity
    value: int | None
    applicability: Literal["applicable", "not_applicable", "unknown"]
    source: str

    @model_validator(mode="after")
    def _value_matches_applicability(self) -> Self:
        if (self.value is not None) is not (self.applicability == "applicable"):
            raise ValueError("a capacity value is present exactly when it is applicable")
        return self


class CapacityReportV2(_Payload):
    schema_version: Literal["capacity-report.v2"] = "capacity-report.v2"
    compiled_design: ArtifactRef
    dimensions: tuple[CapacityValueV2, ...]
    compatible_template_ids: tuple[Identity, ...]
    issues: tuple[ValidationIssueV2, ...]
    status: Literal["pass", "fail"]


class GraphViewSetV2(_Payload):
    schema_version: Literal["graph-view-set.v2"] = "graph-view-set.v2"
    base_view: ArtifactRef
    alternative_views: tuple[ArtifactRef, ...]
    accessible_summaries: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=4000)], ...], Field(min_length=1)]


class DesignReviewBundleV2(_Payload):
    schema_version: Literal["design-review-bundle.v2"] = "design-review-bundle.v2"
    compiled_design: ArtifactRef
    diagnostic_report: ArtifactRef
    capacity_report: ArtifactRef
    graph_views: ArtifactRef
    rejected_methods: dict[str, str]
    assumptions: tuple[str, ...]
    identification_risks: tuple[str, ...]
    sensitivity_requirements: tuple[str, ...]


class DesignApprovalV2(_Payload):
    schema_version: Literal["design-approval.v2"] = "design-approval.v2"
    decision: Literal["approved", "changes_requested", "declined"]
    design_revision: Annotated[int, Field(ge=1)]
    review_bundle: ArtifactRef
    approved_bundle_hash: Sha256Hex | None
    change_requests: tuple[str, ...]

    @model_validator(mode="after")
    def _decision_payload(self) -> Self:
        approved = self.decision == "approved"
        changed = self.decision == "changes_requested"
        if (self.approved_bundle_hash is not None) is not approved:
            raise ValueError("approved_bundle_hash is present exactly for approval")
        if bool(self.change_requests) is not changed:
            raise ValueError("change_requests are present exactly when changes are requested")
        if approved and self.approved_bundle_hash != self.review_bundle.content_hash:
            raise ValueError("approval must bind the exact review bundle hash")
        return self


class DesignOutcomeV2(_Payload):
    schema_version: Literal["design-outcome.v2"] = "design-outcome.v2"
    status: Literal["approved", "needs_context", "needs_data", "unsupported",
                    "changes_requested", "declined", "system_failure"]
    design_revision: Annotated[int, Field(ge=1)]
    compiled_design: ArtifactRef | None
    diagnostic_report: ArtifactRef | None
    capacity_report: ArtifactRef | None
    review_bundle: ArtifactRef | None
    approval: ArtifactRef | None
    issues: tuple[ValidationIssueV2, ...]

    @model_validator(mode="after")
    def _approved_has_complete_handoff(self) -> Self:
        refs = (self.compiled_design, self.diagnostic_report, self.capacity_report,
                self.review_bundle, self.approval)
        if (self.status == "approved") is not all(refs):
            raise ValueError("only an approved outcome has the complete V2 handoff")
        return self
