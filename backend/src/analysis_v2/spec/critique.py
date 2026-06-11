"""Critic contracts: the plan gate (S5/S6) and the claim critique (S9).

The plan critic decides whether execution may start and whether a human
must confirm. The claim critic prevents overclaiming: it labels claim
strength and pins the language the report is allowed to use.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class PlanGateStatus(StrEnum):
    PASS_AUTO_APPROVED = "pass_auto_approved"
    NEEDS_USER_CONFIRMATION = "needs_user_confirmation"
    FAIL_MISSING_REQUIRED_INFO = "fail_missing_required_info"


class ConfirmationItem(BaseModel):
    """One thing the user is asked to confirm or supply."""

    field: str = Field(min_length=1, max_length=120)  # e.g. "cutoff_value"
    label: str = Field(min_length=1, max_length=200)
    current_value: str | None = None
    options: list[str] = Field(default_factory=list)
    required: bool = True
    why: str = Field(min_length=1, max_length=500)


class ConfirmationCard(BaseModel):
    headline: str = Field(min_length=1, max_length=200)
    plan_summary: str = Field(min_length=1, max_length=2000)
    items: list[ConfirmationItem] = Field(default_factory=list)


class PlanCritique(BaseModel):
    status: PlanGateStatus
    reasons: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    confirmation_card: ConfirmationCard | None = None
    summary: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def _coherent(self) -> "PlanCritique":
        if (
            self.status == PlanGateStatus.NEEDS_USER_CONFIRMATION
            and self.confirmation_card is None
        ):
            raise ValueError("needs_user_confirmation requires a confirmation card")
        if self.status == PlanGateStatus.FAIL_MISSING_REQUIRED_INFO and not self.missing_required:
            raise ValueError("fail_missing_required_info must list the missing fields")
        return self


class ClaimStrength(StrEnum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    EXPLORATORY = "exploratory"
    NOT_SUPPORTED = "not_supported"


class ClaimCritique(BaseModel):
    strength: ClaimStrength
    allowed_language: list[str] = Field(default_factory=list)
    forbidden_language: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=2000)
