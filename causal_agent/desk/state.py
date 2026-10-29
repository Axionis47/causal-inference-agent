"""Ephemeral routing inputs. Conversation checkpoints contain only operation references."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated

from typing_extensions import TypedDict

from causal_agent.common.contracts import DesignBrief, FamilyDecision, FamilyVerdict, Handoff, PrefilterVote, QuestionFrame, RunRecord, Thought


@dataclass
class Context:
    """Runtime context, set per invocation and never in state. Empty today; the families are read from their registry."""


class RouteState(TypedDict, total=False):
    """The routing keys: from a question and a memory to a hand-off."""

    runs: list[RunRecord]
    question: str
    context_family: str | None
    matrix: object | None
    fork: object | None
    dataset: str
    prefilter_votes: Annotated[list[PrefilterVote], operator.add]
    frame: QuestionFrame | None
    focus: list[str]  # the families the person said they only care about; empty means every one is in play
    family_verdicts: list[FamilyVerdict]  # by code, from the fit over the memory
    probes: list  # ProbeResult
    fit_status: dict | None
    decision: FamilyDecision | None
    gate_errors: list[str]
    decide_attempts: int
    brief: DesignBrief | None  # the Designer's brief: what the design rests on, in the dataset's terms
    handoff: Handoff | None
    decision_record: str
    specialist_result: dict | None
    debug: Annotated[list[Thought], operator.add]
    memory: object | None  # records.Memory as last saved: a mirror for a watcher such as Studio, written by the nodes, read by none


class PrefilterTask(TypedDict):
    """Input to one prefilter worker. Not the parent state."""

    question: str
    changes: str
    column: str
    card: str
