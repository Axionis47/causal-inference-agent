"""Typed output for the ps_diagnostics agent.

The agent currently writes a loose dict to state.ps_diagnostics. This
typed model freezes the shape so the S19 state-slim pass has a target
to flip the field to; the agent and its dict-style readers (notebook
section, critique agent) stay untouched for now.
"""

from typing import Any

from pydantic import BaseModel, Field


class PSDiagnostics(BaseModel):
    """Propensity score diagnostics: overlap, balance, calibration."""

    model_quality: str = "unknown"  # good / acceptable / poor / unknown
    overlap_percentage: float | None = None
    extreme_propensity_count: int = 0
    max_smd_unweighted: float | None = None
    max_smd_weighted: float | None = None
    calibration_brier_score: float | None = None
    positivity_violations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    # The agent attaches per-tool diagnostic payloads under this key today.
    details: dict[str, Any] = Field(default_factory=dict)
