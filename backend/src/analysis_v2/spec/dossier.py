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


# Roles that must never enter the adjustment set: structural columns
# (treatment / outcome / instrument / time / group), columns whose adjustment
# would bias the estimate (post-treatment, mediator, leakage), or columns that
# are never covariates (identifier). PRE_TREATMENT is the safe adjustment label
# and UNCLEAR is left untouched. design_detection (S3) folds the role table into
# the spec's candidate_confounders using this set; plan_builder (S5) re-applies
# it as a defensive assertion, so the two stages read one source and cannot
# disagree about what is adjusted on.
BANNED_ADJUSTMENT_ROLES: frozenset[RoleLabel] = frozenset(
    {
        RoleLabel.TREATMENT,
        RoleLabel.OUTCOME,
        RoleLabel.POST_TREATMENT,
        RoleLabel.MEDIATOR,
        RoleLabel.INSTRUMENT,
        RoleLabel.TIME,
        RoleLabel.GROUP,
        RoleLabel.IDENTIFIER,
        RoleLabel.LEAKAGE,
    }
)


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
