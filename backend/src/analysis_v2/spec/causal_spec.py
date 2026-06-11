"""CausalSpec: the structured form of the user's free-form causal question.

Produced by the intake agent (S1), refined by design detection (S3), frozen
for execution after the plan gate (S6). Column fields hold names that exist
in the dataset schema; when intake can only guess from natural language it
records candidates instead and flags the field in `missing_info`. Inventing
columns is a contract violation, not a low-confidence answer.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class QuestionType(StrEnum):
    """The 16-type representative taxonomy (manifest `taxonomy` keys)."""

    SIMPLE_EFFECT = "simple_effect"
    BINARY_TREATMENT = "binary_treatment"
    DOSE_RESPONSE = "dose_response"
    MULTI_FACTOR = "multi_factor"
    INTERACTION = "interaction"
    MEDIATION = "mediation"
    DID = "did"
    RDD = "rdd"
    IV = "iv"
    TIME_SERIES_INTERVENTION = "time_series_intervention"
    SURVIVAL = "survival"
    BEFORE_AFTER = "before_after"
    HETEROGENEOUS_EFFECTS = "heterogeneous_effects"
    DRIVER_ANALYSIS = "driver_analysis"
    NO_EFFECT = "no_effect"
    MECHANISM_SEARCH = "mechanism_search"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VariableRef(BaseModel):
    """A reference to a dataset column, or candidates when unresolved.

    `column` is set only when the name exists in the dataset schema.
    `candidates` carries schema columns that plausibly match; `clue` keeps
    the natural-language phrase the reference came from. `derived` marks a
    variable the analysis must construct (e.g. post-period indicator).
    """

    column: str | None = None
    candidates: list[str] = Field(default_factory=list)
    clue: str | None = Field(default=None, max_length=300)
    derived: bool = False

    @property
    def resolved(self) -> bool:
        return self.column is not None


class CausalSpec(BaseModel):
    question_type: QuestionType
    type_candidates: list[QuestionType] = Field(default_factory=list)
    confidence: Confidence = Confidence.LOW

    outcome: VariableRef = Field(default_factory=VariableRef)
    treatment: VariableRef = Field(default_factory=VariableRef)
    mediator: VariableRef | None = None
    moderator: VariableRef | None = None
    instrument: VariableRef | None = None
    running_variable: VariableRef | None = None
    cutoff_value: float | None = None
    time_column: VariableRef | None = None
    group_column: VariableRef | None = None
    event_column: VariableRef | None = None
    duration_column: VariableRef | None = None
    intervention_date: str | None = None  # ISO date for time-series designs
    candidate_factors: list[str] = Field(default_factory=list)  # multi-factor X1..Xn
    candidate_confounders: list[str] = Field(default_factory=list)

    treatment_is_continuous: bool = False
    missing_info: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=2000)
