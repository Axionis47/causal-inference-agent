"""Existing public reply wire contract, shared by chat and analysis result views."""

from pydantic import Field

from backend.context.contracts import BriefDraft, ContextEdit
from backend.validation import Contract, ReplyNumber


class Reply(Contract):
    text: str = Field(min_length=1, max_length=12000)
    context_keys: list[str] = Field(default_factory=list, max_length=20)
    analysis_references: list[str] = Field(
        default_factory=list,
        max_length=100,
        description="Exact references from supplied analysis_references, separate from context-pack keys. These bind saved analysis decisions or numerical evidence to this turn.",
    )
    numbers: list[ReplyNumber] = Field(default_factory=list, max_length=100)
    context_edits: list[ContextEdit] = Field(default_factory=list, max_length=10)
    analysis_brief: BriefDraft | None = Field(default=None, exclude_if=lambda value: value is None, description="An unsaved proposed analysis brief. Only explicit Apply changes writes it atomically with context_edits. Supporting fields select exact context statement keys or edit:<index>; never copy factual claims or set readiness.")
    proposal_passage: str = Field(default="", max_length=8000)


