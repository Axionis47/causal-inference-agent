"""AnalysisState gate-related additions: AWAITING_APPROVAL status + human_approval slot."""
from __future__ import annotations

from src.analysis.agents.base.state import AnalysisState, DatasetInfo, JobStatus
from src.domain.approval import ApprovalDecision, DagEdit, HumanApproval


def _base_state() -> AnalysisState:
    return AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(url="kaggle.com/x"),
    )


# --- JobStatus enum addition ------------------------------------------------


def test_awaiting_approval_status_is_registered():
    assert JobStatus.AWAITING_APPROVAL.value == "awaiting_approval"


def test_awaiting_approval_sits_between_discovery_and_estimation():
    # Ordering of declaration matters for the dispatch focus & UI; if a
    # future edit moves AWAITING_APPROVAL out of place, this test fails
    # loudly. StrEnum preserves declaration order.
    members = list(JobStatus)
    discovery_idx = members.index(JobStatus.DISCOVERING_CAUSAL)
    awaiting_idx = members.index(JobStatus.AWAITING_APPROVAL)
    estimating_idx = members.index(JobStatus.ESTIMATING_EFFECTS)
    assert discovery_idx < awaiting_idx < estimating_idx


# --- AnalysisState.human_approval slot --------------------------------------


def test_human_approval_defaults_to_none():
    state = _base_state()
    assert state.human_approval is None


def test_human_approval_accepts_approve_factory_value():
    state = _base_state()
    state.human_approval = HumanApproval.approve(
        granted_by="analyst@example.com",
        dag_edits=DagEdit(adjustment_set=["age", "education"]),
        appended_context="ignore re74",
    )
    assert state.human_approval.decision == ApprovalDecision.APPROVED
    assert state.human_approval.dag_edits.adjustment_set == ["age", "education"]


def test_state_with_human_approval_round_trips_through_json():
    state = _base_state()
    state.human_approval = HumanApproval.approve(granted_by="r")
    dumped = state.model_dump(mode="json")
    restored = AnalysisState.model_validate(dumped)
    assert restored.human_approval is not None
    assert restored.human_approval.decision == ApprovalDecision.APPROVED
    assert restored.human_approval.granted_by == "r"


def test_state_without_human_approval_round_trips_through_json():
    state = _base_state()
    dumped = state.model_dump(mode="json")
    restored = AnalysisState.model_validate(dumped)
    assert restored.human_approval is None
