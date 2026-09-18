"""Router state. Holds the dataset name and each step's artifacts, never the pack payload."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated

from typing_extensions import TypedDict

from causal_agent.common.contracts import FamilyDecision, FamilyVerdict, Handoff, PrefilterVote, QuestionFrame, Thought


@dataclass
class Context:
    """Runtime context: where knowledge lives. Set per invocation, never in state."""

    registry_path: str | None = None
    width_budget: int = 150


class RouterState(TypedDict, total=False):
    question: str
    dataset: str
    prefilter_votes: Annotated[list[PrefilterVote], operator.add]
    frame: QuestionFrame | None
    family_verdicts: Annotated[list[FamilyVerdict], operator.add]
    decision: FamilyDecision | None
    gate_errors: list[str]
    decide_attempts: int
    handoff: Handoff | None
    decision_record: str
    specialist_result: dict | None
    debug: Annotated[list[Thought], operator.add]


class PrefilterTask(TypedDict):
    """Input to one prefilter worker. Not the parent state."""

    question: str
    changes: str
    column: str
    card: str


class FamilyTask(TypedDict):
    """Input to one family worker. Not the parent state."""

    question: str
    family: str
    family_text: str
    intent: str
    outcome: str
    cause: str
    scope: str
    relevant: str
    digest: str
    cards: str
