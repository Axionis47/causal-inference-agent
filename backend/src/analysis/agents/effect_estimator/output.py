"""Typed output for the effect_estimator agent."""

from typing import Any

from pydantic import BaseModel, Field


class TreatmentEffectResult(BaseModel):
    """Result of a treatment effect estimation."""

    method: str
    estimand: str  # ATE, ATT, CATE
    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float | None = None
    assumptions_tested: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

    # Which variables were analyzed (for multi-pair analysis)
    treatment_variable: str | None = None
    outcome_variable: str | None = None
