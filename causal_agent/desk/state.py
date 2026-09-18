"""The routing state: the dataset name and each step's artifacts. The memory itself is loaded by the nodes, never held in state."""

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


class RouteState(TypedDict, total=False):
    question: str
    dataset: str
    prefilter_votes: Annotated[list[PrefilterVote], operator.add]
    frame: QuestionFrame | None
    family_verdicts: list[FamilyVerdict]   # by code, from the fit over the memory
    probes: list                           # ProbeResult
    fit_status: dict | None
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

RouterState = RouteState  # the older name
