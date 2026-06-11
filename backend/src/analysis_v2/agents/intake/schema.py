"""The LLM-facing intake draft: flat fields, validated by the boundary.

The model fills column fields with names it saw in the schema listing, or
leaves them null and uses *_clue / *_candidates. validate.py is the
authority: anything that is not a real dataset column is quarantined
there, so an invented name can never reach the CausalSpec.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.analysis_v2.spec import Confidence, QuestionType


class IntakeDraft(BaseModel):
    question_type: QuestionType
    type_candidates: list[QuestionType] = Field(
        default_factory=list, description="other plausible types, most likely first"
    )
    confidence: Confidence

    outcome_column: str | None = None
    outcome_candidates: list[str] = Field(default_factory=list)
    outcome_clue: str | None = None

    treatment_column: str | None = None
    treatment_candidates: list[str] = Field(default_factory=list)
    treatment_clue: str | None = None
    treatment_is_continuous: bool = False
    treatment_derived: bool = Field(
        default=False,
        description="true when treatment must be constructed (e.g. post-period flag)",
    )

    mediator_column: str | None = None
    mediator_clue: str | None = None
    moderator_column: str | None = None
    moderator_clue: str | None = None
    instrument_column: str | None = None
    instrument_clue: str | None = None
    running_variable_column: str | None = None
    running_variable_clue: str | None = None
    cutoff_value: float | None = None
    time_column: str | None = None
    group_column: str | None = None
    event_column: str | None = None
    duration_column: str | None = None
    intervention_date: str | None = Field(
        default=None, description="ISO date if the question names one"
    )

    candidate_factors: list[str] = Field(default_factory=list)
    candidate_confounders: list[str] = Field(default_factory=list)

    assumptions: list[str] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)
    notes: str | None = None
    reasoning_summary: str = Field(
        description="3-6 plain sentences for the analyst: what kind of question this "
        "is, which columns map to which roles, and what is still unclear"
    )
