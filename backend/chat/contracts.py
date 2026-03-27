"""Public chat views and model boundaries. Saved domain records own the transcript."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from backend.responses import Reply
from backend.validation import Contract


class CreateConversation(Contract):
    submission_id: UUID
    title: str = Field(min_length=1, max_length=200)
    context_revision_id: UUID | None = None


class SendTurn(Contract):
    submission_id: UUID
    expected_turn: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=8000)
    artifact_ids: list[UUID] = Field(default_factory=list, max_length=4)
    mode: Literal["chat", "context"] = "chat"


class ConversationView(Contract):
    id: UUID
    version_id: UUID
    context_revision_id: UUID | None
    title: str
    created_at: datetime


class AnalysisFile(Contract):
    analysis_id: UUID
    filename: str = Field(min_length=1, max_length=200)


class Interpretation(Contract):
    analysis_ids: list[UUID] = Field(default_factory=list, max_length=4, description="Saved analyses needed for this discussion. Select exact IDs from the analyses catalogue, separately from chat artifact IDs. No selection is needed for general conversation.")
    analysis_files: list[AnalysisFile] = Field(
        default_factory=list,
        max_length=4,
        description="Files from the supplied saved-analysis catalogue needed to answer this message. Select only exact analysis IDs and filenames. Discussion may inspect code, tables or figures without executing anything.",
    )
    exploration: str = Field(
        default="",
        max_length=8000,
        description="Self-contained natural-language task for the visualization agent, resolving follow-up references from history. Empty for discussion alone.",
    )
    passage: str = Field(
        default="",
        max_length=8000,
        description="Exact passage from the CURRENT message requesting computation or a visualization/refinement. Empty for discussion.",
    )
    artifact_ids: list[UUID] = Field(
        default_factory=list,
        max_length=4,
        description="Earlier artifacts explicitly referred to by this request, using supplied identities only.",
    )


class ExploreAction(Contract):
    action: Literal["inspect_input", "run_code", "inspect_output", "finish"]
    code: str = Field(default="", max_length=100000)
    reference: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def separate_actions(self) -> "ExploreAction":
        if (self.action == "run_code") != bool(self.code.strip()):
            raise ValueError("Only run_code takes a nonempty Python script. finish never executes code.")
        return self


class Exploration(Contract):
    status: Literal["completed", "needs_input"]
    summary: str = Field(min_length=1, max_length=4000)
    step: int = Field(ge=0)
    facts: list[str] = Field(default_factory=list, max_length=200)
    figure: bool = False
    table: str | None = None
    method: str = Field(default="", max_length=4000)
    findings: list[str] = Field(default_factory=list, max_length=30)
    limitations: list[str] = Field(default_factory=list, max_length=30)
    needs: list[str] = Field(default_factory=list, max_length=10)


class ArtifactView(Contract):
    id: UUID
    turn_id: UUID
    record: Exploration
    facts: dict[str, float]
    filenames: list[str]
    created_at: datetime


class ScriptAttempt(Contract):
    step: int
    ok: bool
    stderr: str


class TurnView(Contract):
    id: UUID
    number: int
    text: str
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: str
    calls: int
    reply: Reply | None
    error: str | None
    trace_id: UUID
    retryable: bool
    script_attempts: list[ScriptAttempt]
    created_at: datetime
    mode: Literal["chat", "context"] = "chat"
    context_revision_id: UUID | None = None
    applied_revision_id: UUID | None = None


class ConversationDetail(ConversationView):
    turns: list[TurnView]
    artifacts: list[ArtifactView]
    newer_context_available: bool
