"""Tests for intake contracts (T-007; PRD-001 §7, §8, §3.1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from causal.intake.contracts import (
    AVAILABLE_STATUSES,
    COLUMN_SLOTS,
    DATASET_SLOTS,
    ColumnSemanticsV1,
    DatasetSemanticsV1,
    IntakeSubmissionV1,
    SemanticSlotV1,
    SemanticStatus,
)


def slot(status: SemanticStatus = SemanticStatus.NOT_OFFERED, **kwargs: object) -> SemanticSlotV1:
    return SemanticSlotV1(status=status, **kwargs)  # type: ignore[arg-type]


class TestSemanticSlot:
    def test_all_nine_statuses_exist(self) -> None:
        assert len(SemanticStatus) == 9
        assert len(AVAILABLE_STATUSES) == 3

    def test_evidenced_requires_value_and_evidence(self) -> None:
        good = slot(SemanticStatus.EVIDENCED, value="real earnings", evidence_ids=("ev:1",))
        assert good.available
        with pytest.raises(ValidationError):
            slot(SemanticStatus.EVIDENCED, value="x")
        with pytest.raises(ValidationError):
            slot(SemanticStatus.EVIDENCED, evidence_ids=("ev:1",))

    def test_hypothesis_never_cites_evidence(self) -> None:
        good = slot(SemanticStatus.HYPOTHESIS, value="999 may be a sentinel")
        assert good.available
        with pytest.raises(ValidationError):
            slot(SemanticStatus.HYPOTHESIS, value="x", evidence_ids=("ev:1",))

    @pytest.mark.parametrize(
        "status",
        [SemanticStatus.EMPTY, SemanticStatus.NOT_OFFERED, SemanticStatus.FETCH_FAILED,
         SemanticStatus.UNREADABLE, SemanticStatus.WITHHELD, SemanticStatus.NOT_APPLICABLE],
    )
    def test_unavailable_statuses_carry_no_content(self, status: SemanticStatus) -> None:
        empty = slot(status)
        assert not empty.available
        with pytest.raises(ValidationError):
            slot(status, value="sneaky")


class TestSlotSets:
    def test_dataset_semantics_requires_exactly_seven_slots(self) -> None:
        complete = DatasetSemanticsV1(slots={name: slot() for name in DATASET_SLOTS})
        assert set(complete.slots) == set(DATASET_SLOTS)
        partial = {name: slot() for name in DATASET_SLOTS[:-1]}
        with pytest.raises(ValidationError):
            DatasetSemanticsV1(slots=partial)

    def test_column_semantics_requires_exactly_eight_slots(self) -> None:
        complete = ColumnSemanticsV1(
            column_name="re74", slots={name: slot() for name in COLUMN_SLOTS}
        )
        assert set(complete.slots) == set(COLUMN_SLOTS)
        extra = {name: slot() for name in (*COLUMN_SLOTS, "vibe")}
        with pytest.raises(ValidationError):
            ColumnSemanticsV1(column_name="re74", slots=extra)


class TestSubmission:
    def _submission(self, ref: str) -> IntakeSubmissionV1:
        return IntakeSubmissionV1(
            schema_version="intake-submission.v1",
            question_text="Does the program raise earnings?",
            context_text=None,
            kaggle_ref=ref,
            idempotency_key="key-1",
        )

    def test_owner_slug_accepted(self) -> None:
        assert self._submission("lalonde/nsw").kaggle_ref == "lalonde/nsw"

    def test_url_normalized(self) -> None:
        url = "https://www.kaggle.com/datasets/lalonde/nsw?select=nsw.csv"
        assert self._submission(url).kaggle_ref == "lalonde/nsw"

    @pytest.mark.parametrize(
        "bad", ["", "just-a-slug", "a/b/c", "https://example.com/datasets/a/b", "a b/c"]
    )
    def test_bad_reference_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            self._submission(bad)

    def test_empty_question_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IntakeSubmissionV1(
                schema_version="intake-submission.v1", question_text="",
                context_text=None, kaggle_ref="a/b", idempotency_key="k",
            )
