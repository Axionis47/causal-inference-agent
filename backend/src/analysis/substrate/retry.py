"""Declarative retry policy for substrate-aware agents.

Replaces the hand-rolled loops in BaseAgent.reason() and
BaseAgent.execute_with_tracing(). Each agent owns a RetryPolicy that
declares max_attempts and which exception classes to retry vs. veto.

Criticality is a sibling concept: a critical agent that exhausts
retries causes the orchestrator to abort the job; an advisory agent
that exhausts retries causes the orchestrator to mark its slot null
and continue.

Exception classes are matched by name (strings) walking the MRO. The
policy serialises to JSON for traces; storing class names instead of
type objects keeps the serialisation trivial. The MRO walk handles
subclass matching, so a retry_on entry of "ConnectionError" also
catches "ConnectionResetError".
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Criticality = Literal["critical", "advisory"]


class RetryPolicy(BaseModel):
    """Declarative retry policy. Frozen so a shared instance is safe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = Field(ge=1, le=10)
    retry_on: tuple[str, ...] = Field(default=())
    no_retry_on: tuple[str, ...] = Field(default=())


def _mro_names(exc: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__}


def should_retry(
    exc: BaseException,
    policy: RetryPolicy,
    attempts_so_far: int,
) -> bool:
    """Decide whether to retry exc, given attempts already made.

    Rules, evaluated in order:
    1. attempts_so_far >= max_attempts -> False (budget used)
    2. exception's MRO matches no_retry_on -> False (veto wins)
    3. exception's MRO matches retry_on -> True
    4. otherwise -> False (whitelist semantics)
    """
    if attempts_so_far >= policy.max_attempts:
        return False
    names = _mro_names(exc)
    if names & set(policy.no_retry_on):
        return False
    if names & set(policy.retry_on):
        return True
    return False
