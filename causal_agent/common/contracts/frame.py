"""The question as read and the routing's decision: what the desk writes between the memory and the pack."""

from __future__ import annotations

from pydantic import BaseModel, Field

from causal_agent.common.contracts.base import Candidate, Cited, Intent, Scope


class QuestionFrame(BaseModel):
    intent: Intent
    decision_served: str = Field(description="what decision the answer would inform, in one sentence")
    outcome_candidates: list[Candidate] = Field(description="ranked, best first")
    cause_candidates: list[Candidate] = Field(description="ranked, best first; empty if intent is driver_search or root_cause")
    scope: Scope
    relevant_columns: list[Candidate] = Field(
        description="every column that matters to this question, including the outcome and cause, each with why and a citation"
    )
    reasons: list[Cited] = Field(description="anything else the reader should know about how the question was read")

    @property
    def outcome(self) -> str | None:
        return self.outcome_candidates[0].column if self.outcome_candidates else None

    @property
    def cause(self) -> str | None:
        return self.cause_candidates[0].column if self.cause_candidates else None


class PrefilterVote(Cited):
    column: str
    relevant: bool


class NeedCheck(BaseModel):
    need: str
    met: bool
    cites: list[str] = Field(default_factory=list)
    note: str = Field(default="", description="why met or unmet, one sentence")


class FamilyVerdict(BaseModel):
    family: str
    admissible: bool
    needs: list[NeedCheck]
    concern: str = Field(default="", description="if admissible, the weak_when condition that applies here, or empty")


class Rejection(BaseModel):
    family: str
    reason: str
    cites: list[str] = Field(default_factory=list)


class FamilyDecision(BaseModel):
    admissible: list[str]
    chosen: str = Field(description="one of admissible; the literal string 'none' if no family is admissible")
    chosen_assumption: str = Field(description="the assumption this choice bets on, for this data; 'none' if no family chosen")
    why_over_alternatives: str = Field(description="if more than one family was admissible, why this one; else 'only admissible family'")
    rejected: list[Rejection]
    cites: list[str] = Field(default_factory=list, description="pack addresses supporting the choice; may be empty")
