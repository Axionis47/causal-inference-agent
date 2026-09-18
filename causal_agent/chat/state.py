"""Desk state: the interview's keys, so the subgraph reads and writes the claims, plus the runs and the after-phase."""

from __future__ import annotations

import operator
from typing import Annotated

from causal_agent.chat.contracts import AfterReply, Exchange, RunRecord
from causal_agent.intake.interview.state import InterviewState


class ChatState(InterviewState, total=False):
    runs: list[RunRecord]
    phase: str  # before | after
    exchanges: Annotated[list[Exchange], operator.add]
    after_message: str
    after_turn: int
    after_reply: AfterReply | None
    after_errors: list[str]
    after_attempts: int
    brief: str
