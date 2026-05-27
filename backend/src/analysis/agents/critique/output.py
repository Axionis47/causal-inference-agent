"""Typed output for the critique agent."""

from enum import StrEnum

from pydantic import BaseModel


class CritiqueDecision(StrEnum):
    """Decision from the critique agent."""

    APPROVE = "APPROVE"
    ITERATE = "ITERATE"
    REJECT = "REJECT"


class CritiqueFeedback(BaseModel):
    """Feedback from the critique agent."""

    decision: CritiqueDecision
    iteration: int
    scores: dict[str, int]  # dimension -> score (1-5)
    issues: list[str]
    improvements: list[str]
    reasoning: str
