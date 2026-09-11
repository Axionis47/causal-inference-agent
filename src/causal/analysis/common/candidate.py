"""Pure exploration contracts. No dataset, approval, or numerical imports."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from causal.analysis.common.models import Issue, Model, Scalar

Boundary = Literal["design", "data_preflight", "execution"]
RequirementStatus = Literal["satisfied", "unresolved", "violated", "inapplicable"]
Eligibility = Literal["available", "conditional", "blocked", "unsupported"]
Relation = Literal["offers", "requires", "reveals", "excludes", "checked_by"]
ResolutionKind = Literal[
    "study_fact", "scientific_assumption", "scientific_frame", "configuration",
    "role_binding", "data_preparation", "execution", "request",
]


class CandidateOutcome(Model):
    column: str | None = None
    kind: str | None = None
    units: str | None = None
    meaning: str | None = None


class CandidateAssertion(Model):
    name: str
    value: Scalar = None
    evidence: tuple[str, ...] = ()
    support: Literal["fact", "assumption"] = "fact"
    scope: tuple[str, ...] = ()
    original_answer: str | None = None


class VariableBinding(Model):
    role: str
    column: str
    source_references: tuple[str, ...] = ()
    expected_alias: str | None = None
    meaning: str | None = None


class CandidateDraft(Model):
    schema_version: Literal["analysis-candidate.v1"] = "analysis-candidate.v1"
    method: str | None = None
    population: str | None = None
    outcome: CandidateOutcome | None = None
    estimand: str | None = None
    unit_grain: str | None = None
    population_policy: str | None = None
    missingness_policy: str | None = None
    bindings: tuple[VariableBinding, ...] = ()
    facts: tuple[CandidateAssertion, ...] = ()
    configuration: dict[str, Any] = Field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()
    sensitivities: tuple[str, ...] = ()
    seed: int | None = Field(default=None, strict=True)
    candidate_reference: str | None = None
    context_reference: str | None = None
    source_dataset_reference: str | None = None


class ResolutionTarget(Model):
    kind: ResolutionKind
    field: str
    requirement_id: str


class RequirementResult(Model):
    requirement_id: str
    status: RequirementStatus
    boundary: Boundary = "design"
    explanation: str
    fields: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    missing_inputs: tuple[str, ...] = ()
    issues: tuple[Issue, ...] = ()
    resolution_targets: tuple[ResolutionTarget, ...] = ()


class Selection(Model):
    field: str
    value: Any
    origin: Literal["explicit", "fixed_policy", "mechanical_default"]
    node_id: str


class OptionState(Model):
    node_id: str
    state: Eligibility
    selected: bool = False
    explanation: str
    dependencies: tuple[str, ...] = ()
    missing_inputs: tuple[str, ...] = ()
    invalidated: bool = False
    applicability: Literal["applicable", "inapplicable", "unresolved"] = "unresolved"


class RoleSlot(Model):
    node_id: str
    role: str
    field: str
    kind: str
    minimum: int
    maximum: int | None
    bound_columns: tuple[str, ...] = ()
    source_references: tuple[str, ...] = ()
    description: str


class CheckObligation(Model):
    node_id: str
    boundary: Boundary
    selected: bool
    applicability: Literal["applicable", "inapplicable", "unresolved"]
    explanation: str
    dependencies: tuple[str, ...] = ()


class CandidateEvaluation(Model):
    schema_version: Literal["analysis-evaluation.v1"] = "analysis-evaluation.v1"
    candidate_fingerprint: str
    capability_version: str | None
    status: Literal["needs_information", "rejected", "design_ready"]
    requirements: tuple[RequirementResult, ...]
    selections: tuple[Selection, ...] = ()
    options: tuple[OptionState, ...] = ()
    role_slots: tuple[RoleSlot, ...] = ()
    issues: tuple[Issue, ...] = ()
    obligations: tuple[CheckObligation, ...] = ()
    configuration: dict[str, Any] | None = None


class GraphNode(Model):
    node_id: str
    type: Literal["root", "method", "decision", "option", "role", "requirement", "policy", "check", "exclusion"]
    title: str
    description: str
    capability_version: str
    fields: tuple[str, ...] = ()
    permitted_values: tuple[Any, ...] = ()
    input_schema: dict[str, Any] = Field(default_factory=dict)
    boundary: Boundary | None = None
    evidence_expectation: str | None = None
    minimum: int | None = None
    maximum: int | None = None
    applicability_boundary: Boundary | None = None
    measurement_boundary: Boundary | None = None
    requirement_kind: str | None = None
    expected_condition: str | None = None
    failure_category: str | None = None
    explanation_reference: str | None = None


class GraphEdge(Model):
    source: str
    target: str
    relation: Relation


class CapabilityGraph(Model):
    schema_version: Literal["analysis-graph.v1"] = "analysis-graph.v1"
    version: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]


class GraphCoverage(Model):
    total_neighbors: int
    returned_neighbors: int
    offset: int
    truncated: bool
    next_cursor: str | None


class CapabilityExploration(Model):
    schema_version: Literal["analysis-exploration.v1"] = "analysis-exploration.v1"
    graph_version: str
    candidate_fingerprint: str
    at: GraphNode
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    options: tuple[OptionState, ...]
    requirements: tuple[RequirementResult, ...]
    selections: tuple[Selection, ...]
    status: Literal["needs_information", "rejected", "design_ready"]
    blocker_count: int
    blocker_references: tuple[str, ...]
    coverage: GraphCoverage


class CapabilityRequestError(ValueError):
    def __init__(self, field: str, message: str) -> None:
        self.target = ResolutionTarget(kind="request", field=field, requirement_id="analysis:request")
        super().__init__(message)
