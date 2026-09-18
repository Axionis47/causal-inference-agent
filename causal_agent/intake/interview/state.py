"""Interview state. The claim table is the state; the file and its profile are cached by path, never carried."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from causal_agent.common.contracts import Thought
from causal_agent.memory.claims import ClaimTable, ProbeResult, Reply, Status


class Message(TypedDict):
    role: str  # user | assistant
    turn: int
    text: str


class InterviewState(TypedDict, total=False):
    dataset: str
    csv: str
    docs: dict[str, str]  # name -> text given up front (the description); cited as doc:<name>
    question: str | None
    turn: int
    last_message: str
    last_source: str  # doc:<name> or user:turn:<n>
    messages: Annotated[list[Message], operator.add]
    claims: ClaimTable
    probes: list[ProbeResult]
    status: Status | None
    reply: Reply | None
    extract_errors: list[str]
    extract_attempts: int
    respond_errors: list[str]
    respond_attempts: int
    run_requested: bool  # the person said run while claims were open; the drafts were taken on their word
    handoff_ready: bool
    written: dict | None
    debug: Annotated[list[Thought], operator.add]
