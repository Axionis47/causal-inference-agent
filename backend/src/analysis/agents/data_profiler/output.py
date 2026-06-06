"""Typed outputs for the data profiler agent.

The agent produces two slots:
  - DataProfile: column types, missing-value counts, candidates for treatment,
    outcome, confounders, instruments; time-dimension and discontinuity hints.
  - TreatmentEncoding: how a multi-level categorical treatment was collapsed or
    label-encoded so downstream methods stay deterministic.

Both stored at AnalysisState.data_profile and AnalysisState.treatment_encoding
respectively.
"""

from pydantic import BaseModel, Field


class DataProfile(BaseModel):
    """Profile of a dataset from the data profiler agent."""

    n_samples: int
    n_features: int
    feature_names: list[str]
    feature_types: dict[str, str]  # feature_name -> type (numeric, categorical, etc.)
    missing_values: dict[str, int]  # feature_name -> count of missing
    numeric_stats: dict[str, dict[str, float]]  # feature_name -> {mean, std, min, max}
    categorical_stats: dict[str, dict[str, int]]  # feature_name -> {value: count}

    # Causal-specific profiling
    treatment_candidates: list[str] = Field(default_factory=list)
    outcome_candidates: list[str] = Field(default_factory=list)
    potential_confounders: list[str] = Field(default_factory=list)
    potential_instruments: list[str] = Field(default_factory=list)
    has_time_dimension: bool = False
    time_column: str | None = None
    discontinuity_candidates: list[str] = Field(default_factory=list)


class TreatmentEncoding(BaseModel):
    """Profiler-determined encoding for categorical treatments.

    When the treatment variable is a multi-level categorical (string),
    the data profiler LLM decides how to encode it. This encoding is stored
    in state and applied deterministically by all downstream methods.
    """

    original_type: str  # "binary", "multi_categorical", "continuous"
    strategy: str  # "none", "label_encode", "collapse_to_binary"
    control_value: str | None = None  # e.g., "No E-Mail"
    value_mapping: dict[str, int] | None = None  # e.g., {"No E-Mail": 0, "Mens E-Mail": 1}
