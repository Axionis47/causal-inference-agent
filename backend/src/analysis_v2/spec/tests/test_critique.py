"""Plan gate and claim critique invariants."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis_v2.spec import (
    ClaimCritique,
    ClaimStrength,
    ConfirmationCard,
    ConfirmationItem,
    PlanCritique,
    PlanGateStatus,
)


def _rdd_card() -> ConfirmationCard:
    return ConfirmationCard(
        headline="Confirm the scholarship cutoff before running RDD",
        plan_summary="Sharp RDD on score with treatment assigned at the cutoff.",
        items=[
            ConfirmationItem(
                field="cutoff_value",
                label="Score cutoff",
                current_value="50",
                why="The cutoff was inferred from the data, not stated by you.",
            )
        ],
    )


def test_needs_user_confirmation_requires_a_card():
    with pytest.raises(ValidationError):
        PlanCritique(status=PlanGateStatus.NEEDS_USER_CONFIRMATION)
    ok = PlanCritique(
        status=PlanGateStatus.NEEDS_USER_CONFIRMATION,
        reasons=["cutoff inferred, not confirmed"],
        confirmation_card=_rdd_card(),
    )
    assert ok.confirmation_card.items[0].field == "cutoff_value"


def test_fail_missing_required_info_must_list_the_missing_fields():
    with pytest.raises(ValidationError):
        PlanCritique(status=PlanGateStatus.FAIL_MISSING_REQUIRED_INFO)
    ok = PlanCritique(
        status=PlanGateStatus.FAIL_MISSING_REQUIRED_INFO,
        missing_required=["time_column"],
    )
    assert ok.missing_required == ["time_column"]


def test_auto_approval_needs_no_card():
    ok = PlanCritique(status=PlanGateStatus.PASS_AUTO_APPROVED)
    assert ok.confirmation_card is None


def test_claim_strength_labels_and_critique_round_trip():
    assert {s.value for s in ClaimStrength} == {
        "strong",
        "moderate",
        "weak",
        "exploratory",
        "not_supported",
    }
    critique = ClaimCritique(
        strength=ClaimStrength.EXPLORATORY,
        allowed_language=["consistent with", "suggests"],
        forbidden_language=["proves", "definitely causes"],
        limitations=["no control group; before-after comparison only"],
        rationale="Driver analysis without identification cannot support causal claims.",
    )
    loaded = ClaimCritique.model_validate(critique.model_dump(mode="json"))
    assert loaded.strength == ClaimStrength.EXPLORATORY
    assert "proves" in loaded.forbidden_language
