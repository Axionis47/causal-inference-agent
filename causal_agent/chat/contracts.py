"""Typed artifacts of the desk. AfterReply is flat so any structured-output backend accepts it."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from causal_agent.intake.interview.contracts import ClaimUpdate


class RunRecord(BaseModel):
    index: int
    dataset: str
    question: str
    family: str | None = None
    specialist: str | None = None
    status: str = "no_handoff"
    run_dir: str | None = None
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


AfterKind = Literal["answer", "revise", "requestion", "done"]


class AfterReply(BaseModel):
    kind: AfterKind = Field(description="answer: reply from the material; revise: the person changed a claim about the world; requestion: the person asks a new causal question of the same data; done: they are finished")
    text: str = Field(description="the message to the person")
    cites: list[str] = Field(default_factory=list, description="material addresses the message rests on")
    numbers: list[NumberStated] = Field(default_factory=list, description="every number stated in the text, with its address")
    claim_updates: list[ClaimUpdate] = Field(default_factory=list, description="for revise: the claim updates the person's words imply")
    question: str | None = Field(default=None, description="for requestion: the new question, in full")


class Exchange(BaseModel):
    turn: int
    user: str
    assistant: str
    kind: str = "answer"
