"""Typed commands, reference focus and durable question identities for chat."""

from typing import Any, Literal

from pydantic import Field, model_validator

from causal_agent.memory.context.contracts import Contract
from causal_agent.memory.context.review import ContextEdit


class Reference(Contract):
    kind: Literal["column", "artifact"]
    id: str = Field(min_length=1, max_length=100)
    element: str | None = Field(default=None, max_length=200)


class Option(Contract):
    id: str
    label: str
    value: Any = None
    unknown: bool = False


class GapQuestion(Contract):
    id: str
    revision_id: str
    target: str
    text: str
    options: list[Option] = Field(default_factory=list)


class ChatCommand(Contract):
    submission_id: str = Field(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    kind: Literal["message", "correct", "answer", "fast_track", "pause", "focus", "reuse", "refresh", "authorize"]
    text: str = Field(default="", max_length=20000)
    revision_id: str
    references: list[Reference] = Field(default_factory=list, max_length=10)
    edits: list[ContextEdit] = Field(default_factory=list, max_length=100)
    question_id: str | None = None
    option_id: str | None = None
    snapshot_id: str | None = None

    @model_validator(mode="after")
    def intent(self):
        if self.kind == "message" and not self.text.strip():
            raise ValueError("A message needs text.")
        if self.kind == "correct" and not self.edits:
            raise ValueError("A correction needs explicit edits.")
        if self.edits and self.kind != "correct":
            raise ValueError("Only correction commands carry edits.")
        if self.kind == "answer" and (not self.question_id or not (self.option_id or self.text.strip())):
            raise ValueError("Choose an answer or provide free text.")
        if self.kind == "authorize" and not self.snapshot_id:
            raise ValueError("Select the analysis review to authorize.")
        if self.kind == "focus" and len(self.references) > 1:
            raise ValueError("Select one conversation focus or clear it.")
        if self.kind == "reuse" and (len(self.references) != 1 or self.references[0].kind != "artifact"):
            raise ValueError("Select one artifact to reuse.")
        return self


def resolve_focus(explicit: list[Reference], focus: Reference | None) -> list[Reference]:
    """Explicit message references take precedence. An inspector read never calls this."""
    return explicit or ([focus] if focus else [])
