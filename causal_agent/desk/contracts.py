"""Typed artifacts of the desk conversation. Model outputs are flat so any structured-output backend accepts them;
everything else is code-owned."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

AskKind = Literal["confirm", "choose", "open", "columns"]


class Ask(BaseModel):
    """The one question the desk asks this turn: which fields it settles, in what form, and why it is asked."""

    addresses: list[str] = Field(description="the field addresses this question settles")
    kind: AskKind
    text: str
    options: list[str] = Field(default_factory=list, description="for choose: the legal answers")
    because: list[str] = Field(default_factory=list, description="the families that need these fields")
    evidence: list[str] = Field(default_factory=list, description="check addresses shown beside the question, when the file refuted an answer")


class FieldUpdate(BaseModel):
    address: str = Field(description="col:<key>.<field> or claim:<kind>.<field>, exactly as listed")
    value: str = Field(description="the value as text; true/false for yes-no fields; a column name for column fields; comma-separated names for column lists")
    said: str = Field(description="the person's own words this rests on, verbatim, short")
    reason: str = Field(default="", description="one sentence: how the words give this value")


class Inference(BaseModel):
    """What one message settles: the fields it fills, the fields the person confirmed as drafted, and the ones they cannot say."""

    updates: list[FieldUpdate] = Field(default_factory=list)
    confirms: list[str] = Field(default_factory=list, description="addresses of drafted fields the person said are right, as they stand")
    unknown: list[str] = Field(default_factory=list, description="addresses the person said they cannot say")
    note: str = Field(default="", description="anything said that fits no field, one sentence, or empty")


class RunRecord(BaseModel):
    index: int
    dataset: str
    question: str
    family: str | None = None
    specialist: str | None = None
    status: str = "no_handoff"
    run_dir: str | None = None
    design_dir: str | None = None
    figures: list[dict] = Field(default_factory=list, description="FigureSpecs the run left behind, the ready-moment figure first")
    what_if: dict[str, str] = Field(default_factory=dict, description="for a what-if design: the fields changed on the fork, address -> value")
    differs: list[str] = Field(default_factory=list, description="the fields that differ from the design before, by address")
    effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    estimator: str | None = None
    decision_record: str = ""
    decision: dict = Field(default_factory=dict)  # chosen, chosen_assumption, why, over: {family: reason}
    specialist_result: dict = Field(default_factory=dict)
    artifacts: dict = Field(default_factory=dict)  # artifacts.json when the run dir holds one


class NumberStated(BaseModel):
    address: str = Field(description="the artifact address the number comes from, exactly as shown in the material")
    value: float = Field(description="the number as you state it in the text")


AfterKind = Literal["answer", "revise", "what_if", "requestion", "done"]


class AfterReply(BaseModel):
    kind: AfterKind = Field(description="answer: reply from the material; revise: the person changed something about the data; what_if: the person asks what the answer would be if the data had been different, without changing what is known; requestion: a new causal question of the same data; done: they are finished")
    text: str = Field(description="the message to the person")
    cites: list[str] = Field(default_factory=list, description="material addresses the message rests on")
    numbers: list[NumberStated] = Field(default_factory=list, description="every number stated in the text, with its address")
    updates: list[FieldUpdate] = Field(default_factory=list, description="for revise and what_if: the field updates the person's words imply")
    question: str | None = Field(default=None, description="for requestion: the new question, in full")
    figure: str | None = Field(default=None, description="the address of a figure in the material to show beside the text (figure:<id>), when one makes the point")


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
    def of(cls, f: Any) -> "Finding":
        return cls(address=f.address, rule=f.rule, passed=f.passed, detail=f.detail, evidence=f.evidence)
