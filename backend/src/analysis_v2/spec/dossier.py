"""The dataset dossier (S2a): what the dataset is, the role every column
plays relative to the question, and the context ledger that names which
identification assumptions rest on data, description, default, or the
user. Downstream agents read the dossier instead of re-deriving raw
context; the role table is what protects the adjustment set."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .causal_spec import Confidence


class RoleLabel(StrEnum):
    TREATMENT = "treatment"
    OUTCOME = "outcome"
    PRE_TREATMENT = "pre_treatment"  # safe adjustment candidate
    POST_TREATMENT = "post_treatment"  # adjusting would bias the estimate
    MEDIATOR = "mediator"
    INSTRUMENT = "instrument"
    TIME = "time"
    GROUP = "group"
    IDENTIFIER = "identifier"  # row ids, indexes; never a covariate
    LEAKAGE = "leakage"  # near-copies of outcome/treatment
    UNCLEAR = "unclear"


class ColumnRole(BaseModel):
    column: str = Field(min_length=1)
    role: RoleLabel
    reason: str = Field(min_length=1, max_length=300)
    confidence: Confidence = Confidence.MEDIUM


class LedgerStatus(StrEnum):
    ESTABLISHED_FROM_DATA = "established_from_data"
    ASSERTED_FROM_DESCRIPTION = "asserted_from_description"
    ASSUMED_DEFAULT = "assumed_default"
    NEEDS_USER = "needs_user"


class ContextLedgerItem(BaseModel):
    assumption: str = Field(min_length=1, max_length=300)
    status: LedgerStatus
    note: str = Field(default="", max_length=300)


class DatasetDossier(BaseModel):
    provenance: str = Field(min_length=1, max_length=2000)
    row_meaning: str | None = Field(default=None, max_length=300)
    roles: list[ColumnRole] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)
    recommended_exclusions: list[str] = Field(default_factory=list)
    context_ledger: list[ContextLedgerItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1, max_length=2000)
    investigated: bool = True  # False on the degraded no-tools path

    def role_of(self, column: str) -> ColumnRole | None:
        for role in self.roles:
            if role.column == column:
                return role
        return None
