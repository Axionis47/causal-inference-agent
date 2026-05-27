"""Typed output for the sensitivity_analyst agent."""

from typing import Any

from pydantic import BaseModel, Field


class SensitivityResult(BaseModel):
    """Result of a sensitivity analysis."""

    method: str
    robustness_value: float
    interpretation: str
    details: dict[str, Any] = Field(default_factory=dict)
