"""The desk state: the dataset name and each step's artifacts. The memory itself lives on disk and is loaded by the nodes,
never held in state."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated

from typing_extensions import TypedDict

from causal_agent.common.contracts import FamilyDecision, FamilyVerdict, Handoff, PrefilterVote, QuestionFrame, Thought
from causal_agent.desk.contracts import AfterReply, Ask, Exchange, Finding, RunRecord


@dataclass
class Context:
    """Runtime context: where knowledge lives. Set per invocation, never in state."""

    registry_path: str | None = None
    width_budget: int = 150


class RouteState(TypedDict, total=False):
    """The routing keys: from a question and a memory to a hand-off."""

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


class DeskState(RouteState, total=False):
    """The whole conversation: the question first, the journey to ready, the run, and the chat after."""

    turn: int
    message: str                 # the person's last message
    invalid: str | None          # why the last question did not pass, or None
    frame_attempts: int
    status: object | None        # memory.claims.Status: the fit
    open: list                   # ops.Open: what is still vague
    findings: list[Finding]      # the checks' verdicts this turn
    refutations: dict[str, int]  # address -> how many times the file refuted the person's answer
    settled_now: list[str]       # addresses written this turn, for the acknowledgement
    ask: Ask | None
    reply: str                   # the desk's message this turn
    design_dir: str | None       # designs/<n>/ once the pack is written
    infer_errors: list[str]
    infer_attempts: int
    run_requested: bool
    ready: bool
    figure: dict | None          # the figure shown at the ready moment, a FigureSpec
    fork: object | None          # a what-if: a copy of the memory the routing and the run read instead of the one on disk
    what_if: dict[str, str]      # the fields changed on the fork, address -> value
    convinced_version: int       # the memory version the decision shown at the ready moment was made on
    phase: str                   # before | after
    runs: list[RunRecord]
    exchanges: Annotated[list[Exchange], operator.add]
    after_reply: AfterReply | None
    after_errors: list[str]
    after_attempts: int
    brief: str


class PrefilterTask(TypedDict):
    """Input to one prefilter worker. Not the parent state."""

    question: str
    changes: str
    column: str
    card: str


RouterState = RouteState  # the older name
