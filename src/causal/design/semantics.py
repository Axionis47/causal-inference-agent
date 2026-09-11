"""Semantic cards, concepts, measurement, and role models (PRD-002 §12; SC §16.2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal, Self, TypedDict

from pydantic import ConfigDict, Field, model_validator

from causal.design.contracts import _Payload, _Row
from causal.shared.contracts import Identity, ReferenceKind, reference_field
from causal.shared.envelope import CausalFrameV1, EpistemicStatus, SupportClass

__all__ = [
    "COLUMN_CARD_SLOTS",
    "CausalContextV1",
    "CausalEdgeV1",
    "ColumnSemanticCardV1", "ColumnSemanticSlotsV1",
    "ConceptStatus",
    "ConceptV1",
    "GraphAlternativeV1",
    "MeasurementLinkV1",
    "MeasurementMapV1",
    "MeasurementRelation",
    "RoleClaimV1",
    "RoleEvidenceV1",
    "RoleLedgerV1",
    "RoleName",
    "SlotAssertionV1",
    "TimingClass",
]

class TimingClass(StrEnum):
    PRE_TREATMENT = "pre_treatment"
    CONCURRENT = "concurrent"
    POST_TREATMENT = "post_treatment"
    UNKNOWN = "unknown"


class MeasurementRelation(StrEnum):
    MEASURES = "measures"
    PROXIES = "proxies"
    DERIVED_PROPOSED = "derived_proposed"


class ConceptStatus(StrEnum):
    OBSERVED = "observed"
    PROXY_MEASURED = "proxy_measured"
    UNMEASURED = "unmeasured"


class RoleName(StrEnum):
    TREATMENT = "treatment"
    OUTCOME = "outcome"
    UNIT_IDENTIFIER = "unit_identifier"
    TIME = "time"
    ASSIGNMENT_VARIABLE = "assignment_variable"
    GROUP = "group"
    CLUSTER = "cluster"
    STRATUM = "stratum"
    RUNNING_VARIABLE = "running_variable"
    CONFOUNDER_CANDIDATE = "confounder_candidate"
    MEDIATOR = "mediator"
    COLLIDER = "collider"
    INSTRUMENT_CANDIDATE = "instrument_candidate"
    EFFECT_MODIFIER = "effect_modifier"
    SELECTION_VARIABLE = "selection_variable"
    PRECISION_COVARIATE = "precision_covariate"
    EXCLUDED_FROM_DESIGN = "excluded_from_design"
    UNKNOWN = "unknown"


class SlotAssertionV1(_Row):
    """One card slot: a value and the epistemic standing that carries it."""

    value: str | None
    status: EpistemicStatus
    evidence_ids: Annotated[tuple[Identity, ...], reference_field(ReferenceKind.EVIDENCE)]

    @model_validator(mode="after")
    def _evidenced_slot_has_support(self) -> Self:
        if self.status is EpistemicStatus.EVIDENCED and not self.evidence_ids:
            raise ValueError("an evidenced slot needs an evidence id")
        return self


class ColumnSemanticSlotsV1(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")  # type: ignore[misc]
    meaning: SlotAssertionV1
    concept: SlotAssertionV1
    entity: SlotAssertionV1
    kind: SlotAssertionV1
    units: SlotAssertionV1
    scale: SlotAssertionV1
    levels: SlotAssertionV1
    encoding: SlotAssertionV1
    timing: SlotAssertionV1
    measurement_window: SlotAssertionV1
    missing_interpretation: SlotAssertionV1
    source_process: SlotAssertionV1


COLUMN_CARD_SLOTS: Final = tuple(ColumnSemanticSlotsV1.__annotations__)


class ColumnSemanticCardV1(_Payload):
    """What one column means, slot by slot (PRD-002 §12.1)."""

    schema_version: Literal["column-semantic-card.v1"] = "column-semantic-card.v1"
    table_name: Identity
    column_name: Annotated[Identity, reference_field(ReferenceKind.COLUMN)]
    display_name: Identity
    concept_id: Annotated[
        Identity | None, reference_field(ReferenceKind.CONCEPT, declaration=True)
    ]
    timing: TimingClass
    slots: ColumnSemanticSlotsV1
    alternatives: tuple[str, ...]
    conflicts: tuple[str, ...]


class MeasurementLinkV1(_Row):
    """How one column stands in for one concept, and when that column was measured.

    `timing` is carried from the column's validated card because PRD-002 §9.5 routes
    "concepts + timing + unresolved requirements" to the role workers; without it they cannot
    tell a pre-treatment covariate from the outcome (D-098).
    """

    concept_id: Annotated[Identity, reference_field(ReferenceKind.CONCEPT)]
    table_name: Identity
    column_name: Annotated[Identity, reference_field(ReferenceKind.COLUMN)]
    relation: MeasurementRelation
    timing: TimingClass
    notes: str


class ConceptV1(_Row):
    """One named concept and how well the data measures it."""

    concept_id: Annotated[
        Identity, reference_field(ReferenceKind.CONCEPT, declaration=True)
    ]
    name: Identity
    description: str
    status: ConceptStatus


class MeasurementMapV1(_Payload):
    """Concepts and the columns that measure them (PRD-002 §12.1)."""

    schema_version: Literal["measurement-map.v1"] = "measurement-map.v1"
    concepts: Annotated[tuple[ConceptV1, ...], Field(min_length=1)]
    links: tuple[MeasurementLinkV1, ...]


class CausalEdgeV1(_Row):
    """One directed mechanism between two concepts (PRD-002 §12.2)."""

    edge_id: Annotated[
        Identity, reference_field(ReferenceKind.GRAPH_EDGE, declaration=True)
    ]
    source_concept_id: Annotated[Identity, reference_field(ReferenceKind.CONCEPT)]
    target_concept_id: Annotated[Identity, reference_field(ReferenceKind.CONCEPT)]
    timeframe: Identity
    mechanism_summary: str
    supporting_evidence_ids: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.EVIDENCE)
    ]
    contrary_evidence_ids: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.EVIDENCE)
    ]
    status: EpistemicStatus
    differing_alternative_ids: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.ALTERNATIVE)
    ]


class GraphAlternativeV1(_Row):
    """A competing edge set kept alongside the selected one."""

    alternative_id: Annotated[
        Identity, reference_field(ReferenceKind.ALTERNATIVE, declaration=True)
    ]
    label: Identity
    edges: tuple[CausalEdgeV1, ...]


class CausalContextV1(_Payload):
    """The selected causal structure and its live alternatives (PRD-002 §12.2)."""

    schema_version: Literal["causal-context.v1"] = "causal-context.v1"
    frame: CausalFrameV1
    concept_ids: Annotated[
        tuple[Identity, ...], reference_field(
            ReferenceKind.CONCEPT, declaration=True, min_length=2
        )
    ]
    edges: tuple[CausalEdgeV1, ...]
    alternatives: tuple[GraphAlternativeV1, ...]
    selection_notes: str


class RoleClaimV1(_Row):
    """One role assignment with its support and its live alternatives (PRD-002 §12.4)."""

    role: RoleName
    concept_id: Annotated[Identity, reference_field(ReferenceKind.CONCEPT)]
    column_refs: Annotated[tuple[Identity, ...], reference_field(ReferenceKind.COLUMN)]
    evidence_ids: Annotated[tuple[Identity, ...], reference_field(ReferenceKind.EVIDENCE)]
    timing: TimingClass
    graph_edge_ids: Annotated[
        tuple[Identity, ...], reference_field(ReferenceKind.GRAPH_EDGE)
    ]
    support_class: SupportClass
    alternatives: tuple[str, ...]
    status: EpistemicStatus
    methods: Annotated[tuple[Identity, ...], reference_field(ReferenceKind.METHOD)]


class RoleLedgerV1(_Payload):
    """Every role claim for one frame; treatment and outcome are mandatory."""

    schema_version: Literal["role-ledger.v1"] = "role-ledger.v1"
    frame: CausalFrameV1
    claims: Annotated[tuple[RoleClaimV1, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _frame_roles_present(self) -> Self:
        anchors = {RoleName.TREATMENT: self.frame.treatment,
                   RoleName.OUTCOME: self.frame.outcome}
        for role, concept_id in anchors.items():
            found = [claim for claim in self.claims if claim.role is role]
            if len(found) != 1 or found[0].concept_id != concept_id:
                raise ValueError(f"{role.value} must bind exactly once to {concept_id}")
        return self


class RoleEvidenceV1(_Payload):
    """One role worker's hypotheses over its assigned scope (PRD-002 §12.4)."""

    schema_version: Literal["role-evidence.v1"] = "role-evidence.v1"
    assigned_scope: Annotated[tuple[Identity, ...], Field(min_length=1)]
    edge_hypotheses: tuple[CausalEdgeV1, ...]
    role_hypotheses: tuple[RoleClaimV1, ...]
    competing_mechanisms: tuple[str, ...]
