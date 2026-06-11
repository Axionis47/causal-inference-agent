"""Structured agent failure, the only way an agent reports it could not finish.

Agents get max 3 internal repair attempts; after that they return one of
these instead of raising. The orchestrator decides what to do from
`next_action`, never from parsing the message.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class FailureType(StrEnum):
    TOOL_ERROR = "tool_error"
    VALIDATION_ERROR = "validation_error"
    MISSING_INPUT = "missing_input"
    EXECUTION_ERROR = "execution_error"


class NextAction(StrEnum):
    ASK_USER = "ask_user"
    FAIL_JOB = "fail_job"
    RETRY = "retry"
    DOWNGRADE_CONFIDENCE = "downgrade_confidence"


class AgentFailure(BaseModel):
    agent: str = Field(min_length=1)
    failure_type: FailureType
    message: str = Field(min_length=1)
    recoverable: bool
    next_action: NextAction
