"""Typed output for the confounder_discovery agent.

The agent currently writes a loose dict to state.confounder_discovery
(read via .get() by ps_diagnostics, the notebook renderer, and
effect_estimator_react). The typed model below freezes the shape so the
S19 state-slim pass can flip the field over.
"""

from typing import Any

from pydantic import BaseModel, Field


class ConfounderRecord(BaseModel):
    """One variable's confounder evidence — used in statistical_fallback_details."""

    variable: str
    corr_with_treatment: float
    corr_with_outcome: float
    balance_p_value: float | None = None
    strength: float
    reasons: list[str] = Field(default_factory=list)
    source: str = "statistical_fallback"


class ConfounderAnalysis(BaseModel):
    """Ranked confounders plus the LLM's adjustment reasoning."""

    ranked_confounders: list[str] = Field(default_factory=list)
    excluded_variables: list[str] = Field(default_factory=list)
    adjustment_strategy: str = ""
    # The investigation log is heterogeneous (one entry per tool call),
    # so dict[str, Any] until a tighter shape is justified.
    investigation_log: list[dict[str, Any]] = Field(default_factory=list)
    statistical_fallback_details: list[ConfounderRecord] | None = None
