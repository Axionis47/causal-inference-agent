"""Deterministic dataset profile (facts only, no LLM).

Produced before the data-review gate by the deterministic profiler. Pure
pandas/numpy, reproducible from the data alone. The role-candidate guesses the
old profiler carried (treatment/outcome/confounder candidates) are intentionally
gone: roles come from the user, not a machine guess.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DataProfile(BaseModel):
    n_samples: int
    n_features: int
    feature_names: list[str]
    feature_types: dict[str, str]  # column -> binary/numeric/categorical/datetime/...
    missing_values: dict[str, int]  # column -> NaN count
    numeric_stats: dict[str, dict[str, float]] = Field(default_factory=dict)
    categorical_stats: dict[str, dict[str, int]] = Field(default_factory=dict)
    has_time_dimension: bool = False
    time_column: str | None = None

    # UI-hint candidate lists. The deterministic profiler leaves these empty
    # (roles come from the user); kept so the dataset view can read them.
    treatment_candidates: list[str] = Field(default_factory=list)
    outcome_candidates: list[str] = Field(default_factory=list)
    potential_confounders: list[str] = Field(default_factory=list)
    potential_instruments: list[str] = Field(default_factory=list)
    discontinuity_candidates: list[str] = Field(default_factory=list)
