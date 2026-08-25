"""Design frame and outcome contracts (PRD-002 §6, §12.3, §14, §17, §18; SC §11)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from causal.design.contracts import (
    REGISTRY_VERSION_KEYS,
    _Payload,
    _require_exact_keys,
    _Row,
)
from causal.design.semantics import ConceptStatus, RoleName
from causal.shared.contracts import ArtifactRef, Identity, Sha256Hex
from causal.shared.envelope import CausalFrameV1, EpistemicStatus

__all__ = [
    "CAPACITY_DIMENSIONS",
    "DESIGN_COUNT_KEYS",
    "REGISTRY_VERSION_KEYS",
    "CapacityStatus",
    "CausalGraphViewV1",
    "DeliveryCapacityCheckV1",
    "DesignOutcomeStatus",
    "DesignOutcomeV1",
    "DiagnosticResultV1",
    "DiagnosticStatus",
    "ExperimentDesignV1",
    "GraphEdgeViewV1",
    "GraphNodeViewV1",
    "PreRepairFeasibilityReportV1",
    "RunnableFrameContractV1",
]

# The closed delivery-capacity cardinality dimensions (SC §11).
CAPACITY_DIMENSIONS: Final = (
    "arms", "contrasts", "subgroups", "cohorts", "periods", "event_times",
    "cutoff_sides", "series", "evidence_items",
)
# The closed DesignOutcome count keys (PRD-002 §6).
DESIGN_COUNT_KEYS: Final = (
    "columns_triaged", "columns_carded", "deferred_columns", "role_tasks", "corrections",
)

_APPROVED_HANDOFF_FIELDS: Final = (
    "experiment_design", "runnable_frame_contract", "causal_graph_view",
    "capacity_check", "approval",
)


class DiagnosticStatus(StrEnum):
    COMPUTED = "computed"
    PARTIAL = "partial"
    NOT_COMPUTABLE = "not_computable"


class DiagnosticResultV1(_Row):
    """One diagnostic computed over the analysis CSV (PRD-002 §14)."""

    diagnostic_id: Identity
    diagnostic_version: Identity
    status: DiagnosticStatus
    csv_artifact: ArtifactRef
    columns_read: tuple[Identity, ...]
    total_rows: Annotated[int, Field(ge=0)]
    used_rows: Annotated[int, Field(ge=0)]
    unused_reason_counts: dict[str, int]
    row_set_hash: Sha256Hex | None
    values: dict[str, float | int | str | bool | None]
    warnings: tuple[str, ...]
    implementation_version: Identity

    @model_validator(mode="after")
    def _used_rows_within_total(self) -> Self:
        if self.used_rows > self.total_rows:
            raise ValueError("used_rows cannot exceed total_rows")
        return self


class PreRepairFeasibilityReportV1(_Payload):
    """Pre-repair feasibility diagnostics for one candidate method (PRD-002 §14)."""

    schema_version: Literal["pre-repair-feasibility-report.v1"] = "pre-repair-feasibility-report.v1"
    method_id: Identity
    results: Annotated[tuple[DiagnosticResultV1, ...], Field(min_length=1)]


class ExperimentDesignV1(_Payload):
    """The frozen design contract downstream stages execute against (PRD-002 §17)."""

    schema_version: Literal["experiment-design.v1"] = "experiment-design.v1"
    causal_question: str
    intended_decision: str
    selected_csv: ArtifactRef
    method_id: Identity
    method_pack_version: Identity
    rejected_methods: dict[str, str]
    frame: CausalFrameV1
    comparator: str
    unit: str
    estimand: Identity
    measurement_map: ArtifactRef
    causal_context: ArtifactRef
    role_ledger: ArtifactRef
    assumptions: tuple[str, ...]
    identification_risks: tuple[str, ...]
    eligibility_rules: tuple[str, ...]
    mandatory_repair_boundaries: tuple[str, ...]
    forbidden_repair_boundaries: tuple[str, ...]
    imputation_eligible_columns: tuple[Identity, ...]
    imputation_forbidden_columns: tuple[Identity, ...]
    deletion_impact_dimensions: tuple[Identity, ...]
    invalidation_conditions: tuple[str, ...]
    required_prerepair_diagnostics: tuple[Identity, ...]
    required_postrepair_diagnostics: tuple[Identity, ...]
    required_visual_evidence: tuple[Identity, ...]
    primary_contrasts: tuple[str, ...]
    multiplicity_policy: str | None
    capacity_check: ArtifactRef | None
    visualization_catalog_version: Identity
    capacity_registry_version: Identity
    sensitivity_requirements: tuple[str, ...]
    registry_versions: dict[str, Identity]

    @model_validator(mode="after")
    def _rejections_and_registry_pins(self) -> Self:
        if len(self.rejected_methods) != 3:
            raise ValueError("rejected_methods must carry exactly the three unselected methods")
        if self.method_id in self.rejected_methods:
            raise ValueError("the selected method_id cannot also be a rejected method")
        _require_exact_keys(self.registry_versions, REGISTRY_VERSION_KEYS, "registry_versions")
        return self


class RunnableFrameContractV1(_Payload):
    """The executable frame preparation and estimation must satisfy (PRD-002 §18)."""

    schema_version: Literal["runnable-frame-contract.v1"] = "runnable-frame-contract.v1"
    selected_csv: ArtifactRef
    output_grain: str
    key_columns: Annotated[tuple[Identity, ...], Field(min_length=1)]
    required_roles: tuple[RoleName, ...]
    allowed_roles: tuple[RoleName, ...]
    forbidden_roles: tuple[RoleName, ...]
    type_constraints: dict[str, Identity]
    uniqueness_constraints: tuple[str, ...]
    eligibility_rules: tuple[str, ...]
    exclusion_reason_vocabulary: tuple[Identity, ...]
    treatment_missingness_rule: str
    outcome_missingness_rule: str
    method_structure: dict[str, Identity]
    imputation_permitted: tuple[Identity, ...]
    imputation_forbidden: tuple[Identity, ...]
    required_missingness_indicators: tuple[Identity, ...]
    deletion_impact_dimensions: tuple[Identity, ...]
    revision_required_conditions: tuple[str, ...]
    feasibility_gates: tuple[str, ...]
    required_final_diagnostics: tuple[Identity, ...]
    estimator_input_schema: Identity
    experiment_design_hash: Sha256Hex


class GraphNodeViewV1(_Row):
    """One rendered concept node (PRD-002 §12.3)."""

    concept_id: Identity
    label: str
    status: ConceptStatus
    roles: tuple[RoleName, ...]


class GraphEdgeViewV1(_Row):
    """One rendered causal edge (PRD-002 §12.3)."""

    edge_id: Identity
    source_concept_id: Identity
    target_concept_id: Identity
    status: EpistemicStatus


class CausalGraphViewV1(_Payload):
    """The reviewable rendering of the selected causal graph (PRD-002 §12.3)."""

    schema_version: Literal["causal-graph-view.v1"] = "causal-graph-view.v1"
    parents: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    nodes: Annotated[tuple[GraphNodeViewV1, ...], Field(min_length=2)]
    edges: tuple[GraphEdgeViewV1, ...]
    selected_alternative_id: Identity | None
    layout_direction: Literal["TB", "LR"]
    renderer_profile: Identity
    legend_text: str
    disclosure_text: str
    spec_hash: Sha256Hex
    svg: str
    accessible_summary: str
    node_edge_table: str
    renderer_version: Identity
    theme_version: Identity
    validator_version: Identity
    validation_status: Identity

    @model_validator(mode="after")
    def _edges_reference_declared_nodes(self) -> Self:
        declared = {node.concept_id for node in self.nodes}
        for edge in self.edges:
            unknown = sorted({edge.source_concept_id, edge.target_concept_id} - declared)
            if unknown:
                raise ValueError(f"edge {edge.edge_id} references undeclared concepts {unknown}")
        return self


class CapacityStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class DeliveryCapacityCheckV1(_Payload):
    """Whether the delivery surface can carry the chosen method's evidence (SC §11)."""

    schema_version: Literal["delivery-capacity-check.v1"] = "delivery-capacity-check.v1"
    method_id: Identity
    method_profile_id: Identity
    cardinalities: dict[str, int]
    required_visual_evidence: tuple[Identity, ...]
    compatible_templates: tuple[Identity, ...]
    template_limits: dict[str, int]
    accessible_table_capacity: Annotated[int, Field(ge=1)]
    execution_concurrency: Annotated[int, Field(ge=1, le=8)]
    render_concurrency: Annotated[int, Field(ge=1, le=8)]
    status: CapacityStatus
    failure_codes: tuple[Identity, ...]
    visualization_catalog_version: Identity
    capacity_registry_version: Identity
    method_registry_version: Identity

    @model_validator(mode="after")
    def _cardinalities_and_failure_codes(self) -> Self:
        _require_exact_keys(self.cardinalities, CAPACITY_DIMENSIONS, "cardinalities")
        if (self.status is CapacityStatus.PASS) is bool(self.failure_codes):
            raise ValueError("failure_codes must be empty exactly when status is pass")
        return self


class DesignOutcomeStatus(StrEnum):
    APPROVED = "approved"
    NEEDS_CONTEXT = "needs_context"
    CHANGES_REQUESTED = "changes_requested"
    DECLINED = "declined"
    REFUSED = "refused"
    FAILED_OBSERVABILITY = "failed_observability"
    FAILED = "failed"


class DesignOutcomeV1(_Payload):
    """The design stage's single terminal record (PRD-002 §6)."""

    schema_version: Literal["design-outcome.v1"] = "design-outcome.v1"
    status: DesignOutcomeStatus
    design_revision: Annotated[int, Field(ge=1)]
    refusal_code: Identity | None
    error_code: Identity | None
    experiment_design: ArtifactRef | None
    runnable_frame_contract: ArtifactRef | None
    causal_graph_view: ArtifactRef | None
    capacity_check: ArtifactRef | None
    approval: ArtifactRef | None
    open_requirement_ids: tuple[Identity, ...]
    clarification_rounds_used: Annotated[int, Field(ge=0, le=2)]
    counts: dict[str, int]

    @model_validator(mode="after")
    def _status_matches_handoffs(self) -> Self:
        if self.status is DesignOutcomeStatus.APPROVED:
            absent = [name for name in _APPROVED_HANDOFF_FIELDS if getattr(self, name) is None]
            if absent:
                raise ValueError(f"an approved outcome requires {absent}")
        if self.status is DesignOutcomeStatus.REFUSED and self.refusal_code is None:
            raise ValueError("a refused outcome requires a refusal_code")
        _require_exact_keys(self.counts, DESIGN_COUNT_KEYS, "counts")
        return self

