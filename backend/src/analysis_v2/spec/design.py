"""Designs, lanes, eligibility, and the frozen method plan.

Design detection (S3) maps spec + profile to candidate designs and a tool
eligibility table. After the plan gate (S6) exactly one lane runs; the
MethodPlan is the deterministic record of what will execute. Lanes whose
required fields are absent are disabled with a stated reason, never
silently enabled.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from .causal_spec import Confidence


class MethodLane(StrEnum):
    OBSERVATIONAL = "observational"
    MATCHING = "matching"
    DID = "did"
    RDD = "rdd"
    IV = "iv"
    TIME_SERIES = "time_series"
    MEDIATION = "mediation"
    SURVIVAL = "survival"


class EligibilityState(StrEnum):
    ENABLED = "enabled"
    CONDITIONAL = "conditional"  # enabled if a stated condition holds
    DISABLED = "disabled"


class LaneEligibility(BaseModel):
    lane: MethodLane
    state: EligibilityState
    reason: str = Field(min_length=1, max_length=500)
    missing_requirements: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _disabled_says_why(self) -> "LaneEligibility":
        if self.state == EligibilityState.DISABLED and not (
            self.missing_requirements or self.reason
        ):
            raise ValueError("disabled lanes must state missing requirements")
        return self


class ToolEligibility(BaseModel):
    lanes: list[LaneEligibility] = Field(default_factory=list)

    def for_lane(self, lane: MethodLane) -> LaneEligibility | None:
        for entry in self.lanes:
            if entry.lane == lane:
                return entry
        return None

    def enabled_lanes(self) -> list[MethodLane]:
        return [
            e.lane
            for e in self.lanes
            if e.state in (EligibilityState.ENABLED, EligibilityState.CONDITIONAL)
        ]


class DesignCandidate(BaseModel):
    lane: MethodLane
    design_label: str = Field(min_length=1, max_length=120)
    confidence: Confidence = Confidence.LOW
    rationale: str = Field(min_length=1, max_length=1000)
    required_fields: dict[str, str] = Field(default_factory=dict)  # role -> column
    missing_requirements: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MethodPlan(BaseModel):
    """The single deterministic lane that runs after confirmation."""

    lane: MethodLane
    estimator: str = Field(min_length=1, max_length=120)
    estimand: str = Field(min_length=1, max_length=60)  # ate / att / late / ...
    outcome: str = Field(min_length=1)
    treatment: str | None = None
    covariates: list[str] = Field(default_factory=list)
    # Lane-specific deterministic settings (cutoff, bandwidth, periods,
    # instrument, duration/event columns, ...). Typed loosely on purpose:
    # each lane validates its own keys at execution start.
    settings: dict[str, object] = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=1000)
