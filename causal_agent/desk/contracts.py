"""Typed artifacts of the desk conversation. Model outputs are flat so any structured-output backend accepts them;
everything else is code-owned."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AskKind = Literal["story", "confirm", "choose", "open"]


class AskItem(BaseModel):
    """One line of a readback as the page can answer it: the field, the draft as the value the memory would accept back, the
    other legal answers when the field has a fixed set, and the one the desk recommends (its own draft)."""

    address: str
    label: str = Field(description="the field in the person's terms: 'assignment, kind' or \"'lunch' when\"")
    value: str = Field(description="the draft, as text the memory accepts back: an option, yes or no, a column name, words")
    options: list[str] = Field(default_factory=list, description="every legal answer when the field has a fixed set; empty for free text")
    recommended: str | None = Field(default=None, description="the answer the desk recommends: its draft")


class Ask(BaseModel):
    """The one question the desk asks this turn: which fields it settles, in what form, and why it is asked."""

    addresses: list[str] = Field(description="the field addresses this question settles")
    kind: AskKind
    text: str
    options: list[str] = Field(default_factory=list, description="for choose: the legal answers")
    recommended: str | None = Field(default=None, description="the option the desk recommends, when it has one; shown first")
    items: list[AskItem] = Field(default_factory=list, description="for confirm: each draft as a line the page can answer on its own")
    because: list[str] = Field(default_factory=list, description="the families that need these fields")
    decision: str = Field(default="", description="the family decision the question serves, when one rests on these fields")
    evidence: list[str] = Field(default_factory=list, description="check addresses shown beside the question, when the file refuted an answer")
    from_lane: bool = Field(default=False, description="a lane asked it back after a run; the turn is remembered as about lane:<address> so it is asked once")


class FieldUpdate(BaseModel):
    address: str = Field(description="col:<key>.<field> or claim:<kind>.<field>, exactly as listed")
    value: str = Field(description="the value as text; true/false for yes-no fields; a column name for column fields; comma-separated names for column lists")
    said: str = Field(description="the person's own words this rests on, verbatim, short")
    reason: str = Field(default="", description="one sentence: how the words give this value")


ReadKind = Literal["update", "explore", "ask", "run", "transform"]


class Reading(BaseModel):
    artifact_refs: list[str] = Field(
        default_factory=list, description="For exploration: exact artifact IDs explicitly referenced in the supplied conversation; do not invent IDs"
    )
    """What the Reader made of one piece of the person's words, a message or a note: what kind of turn it is, the fields it
    fills, the drafted fields the person confirmed, the ones they cannot say, and what else the words did."""

    kind: ReadKind = Field(
        description="what the message does. update: it says something about the data, answers what was asked, says a draft is right, "
        "or says they cannot say; a note is always update. explore: it asks to look at the data (a picture, a table, a count, a "
        "comparison). ask: it asks the desk something. run: it asks to run the analysis, or says yes to the design the desk showed. "
        "transform: it asks to change the table itself (drop or keep rows, add, recode or rename a column). "
        "A message that says anything about the data is update, even when it also asks for a picture or asks something; only update "
        "carries updates, confirms or unknown"
    )
    updates: list[FieldUpdate] = Field(default_factory=list)
    confirms: list[str] = Field(default_factory=list, description="addresses of drafted fields the person said are right, as they stand")
    unknown: list[str] = Field(default_factory=list, description="addresses the person said they cannot say")
    focus: list[str] | None = Field(
        default=None,
        description="the families the person said they only care about, by name as the map listed them, when the message says so; "
        "an empty list when they say every family is back in play; null when the message says nothing about it",
    )
    question: str | None = Field(
        default=None,
        description="what the person asked the desk, in their words, when the message asks something (what a family is, why a question "
        "is asked, what a term means, what the file could answer); null when it asks nothing. Required when kind is ask; may ride "
        "along with update",
    )
    transform: str | None = Field(default=None, description="for transform, or beside updates: what to change in the table, in the person's words")
    explore: str | None = Field(
        default=None,
        description="what the person asked to look at in the data (a picture, a plot, a table, a count, a comparison of columns or arms), "
        "in their words; null when the message asks for none. Required when kind is explore; may ride along with update",
    )
    note: str = Field(default="", description="anything said that fits no field, one sentence, or empty")


class NumberStated(BaseModel):
    address: str = Field(description="the artifact address the number comes from, exactly as shown in the material")
    value: float = Field(description="the number as you state it in the text")


AfterKind = Literal["answer", "revise", "what_if", "requestion", "explore", "transform", "done"]


class AfterReply(BaseModel):
    artifact_refs: list[str] = Field(
        default_factory=list, description="For exploration: exact artifact IDs from the supplied material needed for this request; never invent IDs"
    )
    """The Explainer's reply, before or after the run. Before the run only answer and explore are legal; the gate refuses the rest."""

    kind: AfterKind = Field(
        description="answer: reply from the material; revise: the person changed something about the data; what_if: the person asks what the answer would be if the data had been different, without changing what is known; requestion: a new causal question of the same data; explore: they ask to look at the data or the run, a picture, a table, a count; done: they are finished"
    )
    text: str = Field(description="the message to the person")
    cites: list[str] = Field(default_factory=list, description="material addresses the message rests on")
    numbers: list[NumberStated] = Field(default_factory=list, description="every number stated in the text, with its address")
    updates: list[FieldUpdate] = Field(default_factory=list, description="for revise and what_if: the field updates the person's words imply")
    question: str | None = Field(default=None, description="for requestion: the new question, in full")
    explore: str | None = Field(default=None, description="for explore: what to look at, in the person's words")
    transform: str | None = Field(default=None, description="for transform: what to change in the table, in the person's words")
    figure: str | None = Field(
        default=None, description="the address of a figure in the material to show beside the text (figure:<id>), when one makes the point"
    )


class Exchange(BaseModel):
    turn: int
    user: str
    assistant: str
    kind: str = "answer"


class Finding(BaseModel):
    """A check's verdict on one field, kept in state so the next question can show it."""

    address: str
    rule: str
    passed: bool | None
    detail: str
    evidence: str

    @classmethod
    def of(cls, f: Any) -> Finding:
        return cls(address=f.address, rule=f.rule, passed=f.passed, detail=f.detail, evidence=f.evidence)
