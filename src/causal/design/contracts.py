"""Design selection, manifest, intent, and interrupt payloads (PRD-002 §4–§6, §11; SC §5.1)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causal.shared.contracts import ArtifactRef, Identity, PayloadLocator, Sha256Hex
from causal.shared.envelope import ClaimV1

__all__ = [
    "REGISTRY_VERSION_KEYS",
    "AnswerItemV1",
    "AnswerKind",
    "ApprovalDecision",
    "AvailabilityRowV1",
    "ConceptProposalV1",
    "DesignApprovalDecisionV1",
    "DesignContextManifestV1",
    "DesignIntentV1",
    "InterruptKind",
    "QuestionItemV1",
    "QuestionKind",
    "SelectionSource",
    "StructuralFieldV1",
    "TableSelectionDecisionV1",
    "TableSelectionV1",
    "UserContextAnswerV1",
    "UserQuestionPacketV1",
]

REGISTRY_VERSION_KEYS: Final = (
    "artifact_types", "field_classes", "method_packs", "requirements", "tools",
    "validators", "capacity", "graph", "schema",
)

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
_PositiveInt = Annotated[int, Field(ge=1)]
_NonNegInt = Annotated[int, Field(ge=0)]


def _require_exact_keys(
    mapping: Mapping[str, object], expected: tuple[str, ...], label: str
) -> None:
    """Reject a mapping whose key set differs from a closed vocabulary."""
    if set(mapping) != set(expected):
        missing = sorted(set(expected) - set(mapping))
        extra = sorted(set(mapping) - set(expected))
        raise ValueError(f"{label} key mismatch: missing={missing} extra={extra}")


class _Row(BaseModel):
    """Frozen, strict base for design row and fragment models."""

    model_config = _MODEL_CONFIG


class _Payload(_Row):
    """Base for committed design payloads and CLI-submitted decisions."""

    def canonical_payload(self) -> dict[str, Any]:
        """Canonical-ready dict; `content_hash(self.canonical_payload())` is replay-stable."""
        return self.model_dump(mode="json")


class SelectionSource(StrEnum):
    ONLY_CANDIDATE = "only_candidate"
    USER_DECISION = "user_decision"


class QuestionKind(StrEnum):
    CAUSAL = "causal"
    PREDICTIVE = "predictive"
    DESCRIPTIVE = "descriptive"
    EXPLORATORY = "exploratory"


class InterruptKind(StrEnum):
    TABLE_SELECTION = "table_selection"
    CLARIFICATION = "clarification"
    APPROVAL = "approval"


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    DECLINED = "declined"


class AnswerKind(StrEnum):
    VALUE = "value"
    UNKNOWN = "unknown"


class TableSelectionV1(_Payload):
    """The one analysis CSV this design revision is bound to (PRD-002 §6)."""

    schema_version: Literal["table-selection.v1"] = "table-selection.v1"
    dataset_id: Identity
    logical_name: Identity
    resource_object_locator: PayloadLocator
    resource_sha256: Sha256Hex
    media_type: Literal["text/csv"] = "text/csv"
    candidate_count: _PositiveInt
    selection_source: SelectionSource
    decision_artifact_id: Identity | None


class StructuralFieldV1(_Row):
    """One structural column of the selected table."""

    table_name: Identity
    column_name: Identity
    dtype: Identity
    ordinal: _NonNegInt


class AvailabilityRowV1(_Row):
    """One availability surface row carried forward from intake."""

    scope_kind: Literal["dataset", "table", "column"]
    table_name: Identity | None
    column_name: Identity | None
    field_or_slot_name: Identity
    status: Identity
    evidence_count: _NonNegInt
    json_pointer: str


class DesignContextManifestV1(_Payload):
    """The closed context surface every design task reads from (SC §5.1)."""

    schema_version: Literal["design-context-manifest.v1"] = "design-context-manifest.v1"
    design_revision: _PositiveInt
    question_artifact: ArtifactRef
    intake_outcome_artifact: ArtifactRef
    table_selection_artifact: ArtifactRef
    selected_table: Identity
    structural_inventory: tuple[StructuralFieldV1, ...]
    semantic_available: tuple[AvailabilityRowV1, ...]
    semantic_missing: tuple[AvailabilityRowV1, ...]
    measured_surface: tuple[AvailabilityRowV1, ...]
    provenance_surface: tuple[AvailabilityRowV1, ...]
    retrieval_surfaces: tuple[Identity, ...]
    registry_versions: dict[str, Identity]
    recipient_map: dict[str, tuple[Identity, ...]]

    @model_validator(mode="after")
    def _closed_registry_versions(self) -> Self:
        _require_exact_keys(self.registry_versions, REGISTRY_VERSION_KEYS, "registry_versions")
        return self


class ConceptProposalV1(_Row):
    """A proposed concept and the columns that might carry it."""

    name: Identity
    description: str
    candidate_columns: tuple[Identity, ...]


class DesignIntentV1(_Payload):
    """What the user is asking, restated as a causal frame proposal (PRD-002 §11)."""

    schema_version: Literal["design-intent.v1"] = "design-intent.v1"
    question_kind: QuestionKind
    causal_claim: str
    intended_decision: str
    treatment: ConceptProposalV1
    outcome: ConceptProposalV1
    population: ConceptProposalV1
    comparator: ConceptProposalV1
    unit: ConceptProposalV1
    timeframe: ConceptProposalV1
    candidate_grain: Identity
    mandatory_concepts: tuple[ConceptProposalV1, ...]
    claims: tuple[ClaimV1, ...]


class QuestionItemV1(_Row):
    """One user-facing clarification, always answerable with `unknown`."""

    question_id: Identity
    requirement_ids: Annotated[tuple[Identity, ...], Field(min_length=1)]
    question_text: str
    why_it_matters: str
    blocked_decisions: tuple[Identity, ...]
    expected_answer_schema: Identity
    allow_unknown: Literal[True] = True


class UserQuestionPacketV1(_Payload):
    """One clarification round: at most five questions, at most two rounds."""

    schema_version: Literal["user-question-packet.v1"] = "user-question-packet.v1"
    packet_id: Identity
    design_revision: _PositiveInt
    round_number: Literal[1, 2]
    questions: Annotated[tuple[QuestionItemV1, ...], Field(min_length=1, max_length=5)]


class AnswerItemV1(_Row):
    """One answer; `unknown` is a first-class answer and carries no value."""

    question_id: Identity
    answer_kind: AnswerKind
    value: str | None

    @model_validator(mode="after")
    def _value_matches_kind(self) -> Self:
        if (self.value is None) is not (self.answer_kind is AnswerKind.UNKNOWN):
            raise ValueError("value must be None if and only if answer_kind is unknown")
        return self


class UserContextAnswerV1(_Payload):
    """The user's answers to one question packet (CLI-submitted)."""

    schema_version: Literal["user-context-answer.v1"] = "user-context-answer.v1"
    packet_id: Identity
    answers: Annotated[tuple[AnswerItemV1, ...], Field(min_length=1)]
    provenance: Literal["user"] = "user"


class TableSelectionDecisionV1(_Payload):
    """The user's table choice, bound to the interrupt hash it answers."""

    schema_version: Literal["table-selection-decision.v1"] = "table-selection-decision.v1"
    interrupt_id: Identity
    expected_interrupt_hash: Sha256Hex
    expected_revision: _PositiveInt
    selected_table: Identity
    idempotency_key: Identity


class DesignApprovalDecisionV1(_Payload):
    """The user's approval verdict on a design revision (PRD-002 §6)."""

    schema_version: Literal["design-approval-decision.v1"] = "design-approval-decision.v1"
    interrupt_id: Identity
    expected_interrupt_hash: Sha256Hex
    expected_revision: _PositiveInt
    decision: ApprovalDecision
    approved_artifacts: tuple[ArtifactRef, ...]
    change_requests: tuple[str, ...]
    idempotency_key: Identity

    @model_validator(mode="after")
    def _payload_matches_decision(self) -> Self:
        approved = self.decision is ApprovalDecision.APPROVED
        changes = self.decision is ApprovalDecision.CHANGES_REQUESTED
        if bool(self.approved_artifacts) is not approved:
            raise ValueError("approved_artifacts is non-empty if and only if decision is approved")
        if bool(self.change_requests) is not changes:
            raise ValueError(
                "change_requests is non-empty if and only if decision is changes_requested"
            )
        return self
