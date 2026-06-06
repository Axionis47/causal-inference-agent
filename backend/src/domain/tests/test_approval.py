"""HumanApproval / DagEdit / ApprovalDecision: gate transport object."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.domain.approval import ApprovalDecision, DagEdit, HumanApproval


# --- ApprovalDecision enum --------------------------------------------------


def test_decision_enum_values_are_lowercase_strings():
    assert ApprovalDecision.APPROVED.value == "approved"
    assert ApprovalDecision.REJECTED.value == "rejected"


# --- HumanApproval.approve factory ------------------------------------------


def test_approve_factory_sets_decision_and_tz_aware_now():
    approval = HumanApproval.approve(granted_by="analyst@example.com")
    assert approval.decision == ApprovalDecision.APPROVED
    assert approval.granted_by == "analyst@example.com"
    assert approval.granted_at.tzinfo is not None
    # within last 5 seconds is enough; we're not testing wall-clock precision
    delta = datetime.now(timezone.utc) - approval.granted_at
    assert 0 <= delta.total_seconds() < 5


def test_approve_factory_accepts_dag_edits_and_notes():
    edits = DagEdit(adjustment_set=["age", "education"])
    approval = HumanApproval.approve(
        dag_edits=edits,
        appended_context="ignore re74; it's pre-treatment but unreliable",
    )
    assert approval.dag_edits is not None
    assert approval.dag_edits.adjustment_set == ["age", "education"]
    assert approval.appended_context.startswith("ignore re74")


# --- HumanApproval.reject factory -------------------------------------------


def test_reject_factory_requires_reason_and_records_it():
    approval = HumanApproval.reject(reason="adjustment set looks wrong")
    assert approval.decision == ApprovalDecision.REJECTED
    assert approval.reason == "adjustment set looks wrong"


# --- Rejection without reason fails -----------------------------------------


def test_rejection_without_reason_raises():
    with pytest.raises(ValidationError, match="reason is required"):
        HumanApproval(
            decision=ApprovalDecision.REJECTED,
            granted_at=datetime.now(timezone.utc),
        )


def test_rejection_with_blank_reason_raises():
    with pytest.raises(ValidationError, match="reason is required"):
        HumanApproval(
            decision=ApprovalDecision.REJECTED,
            granted_at=datetime.now(timezone.utc),
            reason="   ",
        )


# --- Approval without reason is fine ----------------------------------------


def test_approval_without_reason_is_valid():
    HumanApproval(
        decision=ApprovalDecision.APPROVED,
        granted_at=datetime.now(timezone.utc),
    )


# --- Timezone awareness -----------------------------------------------------


def test_granted_at_must_be_timezone_aware():
    with pytest.raises(ValidationError, match="timezone-aware"):
        HumanApproval(
            decision=ApprovalDecision.APPROVED,
            granted_at=datetime.utcnow(),  # naive
        )


# --- DagEdit shape ----------------------------------------------------------


def test_dag_edit_defaults_to_all_none():
    edit = DagEdit()
    assert edit.adjustment_set is None
    assert edit.forbidden_edges is None
    assert edit.variable_roles is None


def test_dag_edit_round_trips_full_payload():
    edit = DagEdit(
        adjustment_set=["age", "education", "race"],
        forbidden_edges=[{"source": "re78", "target": "re74", "reason": "temporal"}],
        variable_roles={"age": "confounder", "re78": "outcome"},
    )
    dumped = edit.model_dump()
    restored = DagEdit.model_validate(dumped)
    assert restored == edit


def test_dag_edit_rejects_extra_fields():
    with pytest.raises(ValidationError):
        DagEdit.model_validate({"adjustment_set": ["age"], "bogus": 1})


# --- Full HumanApproval round-trip ------------------------------------------


def test_full_approval_round_trips_through_json_mode():
    approval = HumanApproval.approve(
        granted_by="reviewer",
        dag_edits=DagEdit(adjustment_set=["age"]),
        appended_context="LaLonde sample; race coded 1=black, 2=hispanic",
    )
    dumped = approval.model_dump(mode="json")
    restored = HumanApproval.model_validate(dumped)
    assert restored.decision == approval.decision
    assert restored.granted_by == approval.granted_by
    assert restored.dag_edits == approval.dag_edits
    assert restored.appended_context == approval.appended_context


# --- Appended-context length cap --------------------------------------------


def test_appended_context_caps_at_4000_chars():
    HumanApproval.approve(appended_context="x" * 4000)  # boundary OK
    with pytest.raises(ValidationError):
        HumanApproval.approve(appended_context="x" * 4001)
