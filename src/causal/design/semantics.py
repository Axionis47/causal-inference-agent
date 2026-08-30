"""Semantic cards, concepts, measurement, and role models (PRD-002 §12; SC §16.2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from causal.design.contracts import _Payload, _require_exact_keys, _Row
from causal.shared.contracts import Identity
from causal.shared.envelope import CausalFrameV1, ClaimV1, EpistemicStatus, SupportClass

__all__ = [
    "COLUMN_CARD_SLOTS",
    "CausalContextV1",
    "CausalEdgeV1",
    "ColumnSemanticCardV1",
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

COLUMN_CARD_SLOTS: Final = (
    "meaning", "concept", "entity", "kind", "units", "scale", "levels", "encoding",
    "timing", "measurement_window", "missing_interpretation", "source_process",
)


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
    evidence_ids: tuple[Identity, ...]


class ColumnSemanticCardV1(_Payload):
    """What one column means, slot by slot (PRD-002 §12.1)."""

    schema_version: Literal["column-semantic-card.v1"] = "column-semantic-card.v1"
    table_name: Identity
    column_name: Identity
    display_name: Identity
    concept_id: Identity | None
    timing: TimingClass
    slots: dict[str, SlotAssertionV1]
    claims: tuple[ClaimV1, ...]
    alternatives: tuple[str, ...]
    conflicts: tuple[str, ...]

    @model_validator(mode="after")
    def _closed_slots(self) -> Self:
        _require_exact_keys(self.slots, COLUMN_CARD_SLOTS, "slots")
        return self


class MeasurementLinkV1(_Row):
    """How one column stands in for one concept, and when that column was measured.

    `timing` is carried from the column's validated card because PRD-002 §9.5 routes
    "concepts + timing + unresolved requirements" to the role workers; without it they cannot
    tell a pre-treatment covariate from the outcome (D-098).
    """

    concept_id: Identity
    table_name: Identity
    column_name: Identity
    relation: MeasurementRelation
    timing: TimingClass
    notes: str


class ConceptV1(_Row):
    """One named concept and how well the data measures it."""

    concept_id: Identity
    name: Identity
    description: str
    status: ConceptStatus


class MeasurementMapV1(_Payload):
    """Concepts and the columns that measure them (PRD-002 §12.1)."""

    schema_version: Literal["measurement-map.v1"] = "measurement-map.v1"
    concepts: Annotated[tuple[ConceptV1, ...], Field(min_length=1)]
    links: tuple[MeasurementLinkV1, ...]
    claims: tuple[ClaimV1, ...]


class CausalEdgeV1(_Row):
    """One directed mechanism between two concepts (PRD-002 §12.2)."""

    edge_id: Identity
    source_concept_id: Identity
    target_concept_id: Identity
    timeframe: Identity
    mechanism_summary: str
    supporting_evidence_ids: tuple[Identity, ...]
    contrary_evidence_ids: tuple[Identity, ...]
    status: EpistemicStatus
    differing_alternative_ids: tuple[Identity, ...]


class GraphAlternativeV1(_Row):
    """A competing edge set kept alongside the selected one."""

    alternative_id: Identity
    label: Identity
    edges: tuple[CausalEdgeV1, ...]


class CausalContextV1(_Payload):
    """The selected causal structure and its live alternatives (PRD-002 §12.2)."""

    schema_version: Literal["causal-context.v1"] = "causal-context.v1"
    frame: CausalFrameV1
    concept_ids: Annotated[tuple[Identity, ...], Field(min_length=2)]
    edges: tuple[CausalEdgeV1, ...]
    alternatives: tuple[GraphAlternativeV1, ...]
    selection_notes: str
    claims: tuple[ClaimV1, ...]

    @model_validator(mode="after")
    def _edges_stay_inside_concepts(self) -> Self:
        known = set(self.concept_ids)
        edges = [*self.edges, *(edge for alt in self.alternatives for edge in alt.edges)]
        for edge in edges:
            unknown = sorted({edge.source_concept_id, edge.target_concept_id} - known)
            if unknown:
                raise ValueError(f"edge {edge.edge_id} cites unknown concept ids: {unknown}")
        return self


class RoleClaimV1(_Row):
    """One role assignment with its support and its live alternatives (PRD-002 §12.4)."""

    role: RoleName
    concept_id: Identity
    column_refs: tuple[Identity, ...]
    evidence_ids: tuple[Identity, ...]
    timing: TimingClass
    graph_edge_ids: tuple[Identity, ...]
    support_class: SupportClass
    alternatives: tuple[str, ...]
    status: EpistemicStatus
    methods: tuple[Identity, ...]


class RoleLedgerV1(_Payload):
    """Every role claim for one frame; treatment and outcome are mandatory."""

    schema_version: Literal["role-ledger.v1"] = "role-ledger.v1"
    frame: CausalFrameV1
    claims: Annotated[tuple[RoleClaimV1, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _frame_roles_present(self) -> Self:
        roles = {claim.role for claim in self.claims}
        missing = sorted(r for r in (RoleName.TREATMENT, RoleName.OUTCOME) if r not in roles)
        if missing:
            raise ValueError(f"role ledger is missing a claim for: {missing}")
        return self


class RoleEvidenceV1(_Payload):
    """One role worker's hypotheses over its assigned scope (PRD-002 §12.4)."""

    schema_version: Literal["role-evidence.v1"] = "role-evidence.v1"
    assigned_scope: Annotated[tuple[Identity, ...], Field(min_length=1)]
    edge_hypotheses: tuple[CausalEdgeV1, ...]
    role_hypotheses: tuple[RoleClaimV1, ...]
    competing_mechanisms: tuple[str, ...]
    claims: tuple[ClaimV1, ...]
