"""Intake contracts: context classes, semantic slots, availability, submission.

PRD-001 §5.7 (classification), §7 (slots), §8 (availability), §3.1
(submission boundary); decisions D-023, D-024.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from causal.shared.contracts import Identity

__all__ = [
    "AVAILABLE_STATUSES",
    "COLUMN_SLOTS",
    "DATASET_SLOTS",
    "ColumnSemanticsV1",
    "ContextClass",
    "DatasetSemanticsV1",
    "IntakeSubmissionV1",
    "SemanticSlotV1",
    "SemanticStatus",
]


class ContextClass(StrEnum):
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    MEASURED = "measured"
    PROVENANCE = "provenance"
    OPERATIONAL = "operational"
    POPULARITY = "popularity"
    WITHHELD = "withheld"


class SemanticStatus(StrEnum):
    EVIDENCED = "evidenced"
    HINTED = "hinted"
    HYPOTHESIS = "hypothesis"
    EMPTY = "empty"
    NOT_OFFERED = "not_offered"
    FETCH_FAILED = "fetch_failed"
    UNREADABLE = "unreadable"
    WITHHELD = "withheld"
    NOT_APPLICABLE = "not_applicable"


AVAILABLE_STATUSES: Final = frozenset(
    {SemanticStatus.EVIDENCED, SemanticStatus.HINTED, SemanticStatus.HYPOTHESIS}
)

DATASET_SLOTS: Final = (
    "analysis_table", "assignment_mechanism", "population", "unit_of_observation",
    "sampling_rule", "time_span", "table_relationships",
)
COLUMN_SLOTS: Final = (
    "meaning", "kind", "units", "levels", "missing_sentinel", "timing",
    "measurement_window", "provenance",
)

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)


class SemanticSlotV1(BaseModel):
    """One semantic slot: the status is the single source of truth (PRD-001 §8)."""

    model_config = _MODEL_CONFIG

    status: SemanticStatus
    value: str | None = None
    evidence_ids: tuple[Identity, ...] = ()

    @property
    def available(self) -> bool:
        return self.status in AVAILABLE_STATUSES

    @model_validator(mode="after")
    def _content_matches_status(self) -> Self:
        if self.status in (SemanticStatus.EVIDENCED, SemanticStatus.HINTED):
            if not self.value or not self.evidence_ids:
                raise ValueError(f"{self.status} requires a value and at least one evidence ID")
        elif self.status is SemanticStatus.HYPOTHESIS:
            if not self.value:
                raise ValueError("hypothesis requires a value")
            if self.evidence_ids:
                raise ValueError("a hypothesis can never cite provider evidence")
        elif self.value is not None or self.evidence_ids:
            raise ValueError(f"unavailable status {self.status} cannot carry content")
        return self


def _require_exact_slots(
    slots: dict[str, SemanticSlotV1], expected: tuple[str, ...]
) -> dict[str, SemanticSlotV1]:
    if set(slots) != set(expected):
        missing = sorted(set(expected) - set(slots))
        extra = sorted(set(slots) - set(expected))
        raise ValueError(f"slot set mismatch: missing={missing} extra={extra}")
    return slots


class DatasetSemanticsV1(BaseModel):
    """Exactly the seven dataset slots; absence of a slot is a schema error."""

    model_config = _MODEL_CONFIG

    slots: dict[str, SemanticSlotV1]

    @model_validator(mode="after")
    def _exact_slots(self) -> Self:
        _require_exact_slots(self.slots, DATASET_SLOTS)
        return self


class ColumnSemanticsV1(BaseModel):
    """Exactly the eight column slots for one admitted column."""

    model_config = _MODEL_CONFIG

    column_name: Identity
    slots: dict[str, SemanticSlotV1]

    @model_validator(mode="after")
    def _exact_slots(self) -> Self:
        _require_exact_slots(self.slots, COLUMN_SLOTS)
        return self


_OWNER_SLUG = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_KAGGLE_URL = re.compile(
    r"^https?://(?:www\.)?kaggle\.com/datasets/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)(?:[/?#].*)?$"
)


class IntakeSubmissionV1(BaseModel):
    """The only input `causal new` accepts (PRD-001 §3.1). Never carries credentials."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["intake-submission.v1"]
    question_text: Annotated[str, StringConstraints(min_length=1, max_length=10_000)]
    context_text: Annotated[str, StringConstraints(min_length=1, max_length=100_000)] | None
    kaggle_ref: str
    idempotency_key: Identity

    @model_validator(mode="after")
    def _normalize_kaggle_ref(self) -> Self:
        ref = self.kaggle_ref.strip()
        url_match = _KAGGLE_URL.match(ref)
        if url_match:
            ref = f"{url_match.group(1)}/{url_match.group(2)}"
        if not _OWNER_SLUG.match(ref):
            raise ValueError(
                "kaggle_ref must be owner/slug or a kaggle.com/datasets URL (D-024)"
            )
        object.__setattr__(self, "kaggle_ref", ref)
        return self
