"""Cross-stage agent task envelope, result, claims (SYSTEM-CONTRACT §5.2, §6.1, §16.2; D-040)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causal.shared.contracts import ArtifactRef, Identity, Sha256Hex

__all__ = [
    "AgentTaskEnvelopeV1",
    "AgentTaskResultV1",
    "AttemptedEvidenceV1",
    "CausalFrameV1",
    "ClaimV1",
    "ContextRequirementV1",
    "Criticality",
    "EpistemicStatus",
    "EvidenceClass",
    "MissingAction",
    "RequirementScopeKind",
    "SupportClass",
    "SupportRequirement",
    "TaskBudgets",
    "TaskStatus",
    "ToolCallStatus",
    "ToolReceiptV1",
]

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)

JsonScalar = str | int | float | bool | None


class TaskStatus(StrEnum):
    COMPLETE = "complete"
    NEEDS_CONTEXT = "needs_context"
    CONFLICT = "conflict"
    REFUSED = "refused"


class EpistemicStatus(StrEnum):
    EVIDENCED = "evidenced"
    HYPOTHESIS = "hypothesis"
    DISPUTED = "disputed"
    UNKNOWN = "unknown"


class SupportClass(StrEnum):
    DIRECT_USER_CONFIRMATION = "direct_user_confirmation"
    DIRECT_SOURCE_STATEMENT = "direct_source_statement"
    CORROBORATED_SOURCE_INFERENCE = "corroborated_source_inference"
    MEASURED_OBSERVATION = "measured_observation"
    MODEL_HYPOTHESIS = "model_hypothesis"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"


class EvidenceClass(StrEnum):
    USER_CONFIRMATION = "user_confirmation"
    DATA_DICTIONARY = "data_dictionary"
    STUDY_PROTOCOL = "study_protocol"
    SOURCE_STATEMENT = "source_statement"
    TIMESTAMP_RELATIONSHIP = "timestamp_relationship"
    MEASURED_OBSERVATION = "measured_observation"


class Criticality(StrEnum):
    BLOCKING = "blocking"
    SUPPORTING = "supporting"


class MissingAction(StrEnum):
    ASK_USER = "ask_user"
    RETAIN_AS_SENSITIVITY = "retain_as_sensitivity"
    REFUSE = "refuse"


class RequirementScopeKind(StrEnum):
    DATASET = "dataset"
    TABLE = "table"
    COLUMN = "column"
    CONCEPT = "concept"
    RELATIONSHIP = "relationship"
    DESIGN = "design"


class SupportRequirement(StrEnum):
    DIRECT = "direct"
    DIRECT_OR_CORROBORATED = "direct_or_corroborated"
    ANY_ACCEPTABLE = "any_acceptable"


class ToolCallStatus(StrEnum):
    COMPLETED = "completed"
    DENIED = "denied"
    FAILED = "failed"


class CausalFrameV1(BaseModel):
    """The four frame anchors every causal claim is stated against."""

    model_config = _MODEL_CONFIG

    treatment: Identity
    outcome: Identity
    population: Identity
    timeframe: Identity


class ClaimV1(BaseModel):
    """One typed assertion with its own support; no global confidence number."""

    model_config = _MODEL_CONFIG

    claim_id: Identity
    subject_kind: Identity
    subject_id: Identity
    predicate: Identity
    value: JsonScalar
    epistemic_status: EpistemicStatus
    supporting_evidence_ids: tuple[Identity, ...]
    contrary_evidence_ids: tuple[Identity, ...]
    support_class: SupportClass
    alternatives: tuple[str, ...]
    causal_frame: CausalFrameV1 | None

    @model_validator(mode="after")
    def _no_self_citation(self) -> Self:
        cited = set(self.supporting_evidence_ids) | set(self.contrary_evidence_ids)
        if self.claim_id in cited:
            raise ValueError("an inference cannot cite itself as evidence")
        return self


class AttemptedEvidenceV1(BaseModel):
    """One evidence source already tried; intake statuses pass through verbatim."""

    model_config = _MODEL_CONFIG

    evidence_id: Identity
    availability_status: Identity


class ContextRequirementV1(BaseModel):
    """One missing fact, what it blocks, and what may satisfy it (SYSTEM-CONTRACT §6.1)."""

    model_config = _MODEL_CONFIG

    requirement_id: Identity
    registry_version: Identity
    scope_kind: RequirementScopeKind
    scope_id: Identity
    fact_required: str
    why_required: str
    decisions_blocked: tuple[Identity, ...]
    criticality: Criticality
    acceptable_evidence_types: tuple[EvidenceClass, ...]
    required_support: SupportRequirement
    methods_required_for: tuple[Identity, ...]
    attempted_evidence: tuple[AttemptedEvidenceV1, ...]
    user_may_know: bool
    expected_answer_schema: Identity
    missing_action: MissingAction


class TaskBudgets(BaseModel):
    """The per-task spend ceilings the coordinator enforces."""

    model_config = _MODEL_CONFIG

    token_budget: Annotated[int, Field(gt=0)]
    tool_call_budget: Annotated[int, Field(ge=0)]
    transient_attempt_budget: Annotated[int, Field(ge=0)] = 3
    correction_budget: Annotated[int, Field(ge=0)] = 2


class ToolReceiptV1(BaseModel):
    """One tool call as the agent saw it end."""

    model_config = _MODEL_CONFIG

    tool_id: Identity
    call_index: Annotated[int, Field(ge=0)]
    status: ToolCallStatus
    error_code: Identity | None


class AgentTaskEnvelopeV1(BaseModel):
    """Everything one model task may see and do (SYSTEM-CONTRACT §5.2)."""

    model_config = _MODEL_CONFIG

    envelope_id: Identity
    schema_version: Literal["agent-task-envelope.v1"]
    analysis_id: Identity
    stage_run_id: Identity
    task_id: Identity
    attempt_id: Identity
    context_manifest: ArtifactRef
    task_kind: Identity
    scope_kind: Identity
    scope_ids: tuple[Identity, ...]
    parent_artifacts: tuple[ArtifactRef, ...]
    allowed_evidence_ids: tuple[Identity, ...]
    allowed_retrieval_ids: tuple[Identity, ...]
    allowed_tool_ids: tuple[Identity, ...]
    output_schema_version: Identity
    validator_version: Identity
    prompt_version: Identity
    model_profile_version: Identity
    budgets: TaskBudgets
    allowed_stopping_states: tuple[TaskStatus, ...]
    error_vocabulary: tuple[Identity, ...]
    forbidden_payload_classes: tuple[Identity, ...]
    payload_type: Identity
    payload: dict[str, object]

    def canonical_payload(self) -> dict[str, Any]:
        """Canonical-ready dict; `content_hash(self.canonical_payload())` is replay-stable."""
        return self.model_dump(mode="json")


class AgentTaskResultV1(BaseModel):
    """Everything one model task returns (SYSTEM-CONTRACT §5.2; PRD-002 §16.1)."""

    model_config = _MODEL_CONFIG

    envelope_id: Identity
    schema_version: Literal["agent-task-result.v1"]
    task_id: Identity
    status: TaskStatus
    artifact_type: Identity
    artifact_schema_version: Identity
    parent_artifact_ids: tuple[Identity, ...]
    payload: dict[str, object]
    claims: tuple[ClaimV1, ...]
    missing_requirements: tuple[ContextRequirementV1, ...]
    conflicts: tuple[str, ...]
    warnings: tuple[str, ...]
    evidence_ids: tuple[Identity, ...]
    tool_receipts: tuple[ToolReceiptV1, ...]
    output_hash: Sha256Hex | None
    validation_target: Identity

    def canonical_payload(self) -> dict[str, Any]:
        """Canonical-ready dict; `content_hash(self.canonical_payload())` is replay-stable."""
        return self.model_dump(mode="json")
