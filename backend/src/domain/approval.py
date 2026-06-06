"""Human-in-the-loop approval at the post-DAG / pre-estimation gate.

The orchestrator pauses after `dag_expert` produces a refined DAG and
before `effect_estimator` is dispatched. A human reviews the data
quality, the proposed DAG, and the sealed agents' briefs, optionally
edits the DAG (adjustment set, forbidden edges, variable roles) and
appends free-text domain context, then approves or rejects.

`HumanApproval` is the transport object that crosses the API boundary
and lands on `AnalysisState.human_approval`. The orchestrator's gate
check treats a present-and-APPROVED record as "proceed"; anything else
means stay parked.

Method selection is deliberately not part of the edit surface — the
effect_estimator picks methods from data conditions (n, overlap,
treatment type) and pre-committing a list creates irreconcilable
conflicts when assumptions fail.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApprovalDecision(StrEnum):
    """Outcome of the human review at the gate."""

    APPROVED = "approved"
    REJECTED = "rejected"


class DagEdit(BaseModel):
    """Optional edits the human applies to `state.refined_dag` at the gate.

    Each field, when present, replaces the corresponding field on the
    DAG. Absent fields leave the agent-produced value untouched. The
    forbidden-edges shape matches `CausalDAG.forbidden_edges`:
    `{"source": str, "target": str, "reason": str}`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    adjustment_set: list[str] | None = None
    forbidden_edges: list[dict[str, str]] | None = None
    variable_roles: dict[str, str] | None = None


class HumanApproval(BaseModel):
    """One human review decision at the approval gate.

    Persisted onto `AnalysisState.human_approval`. The orchestrator's
    gate check treats `decision == APPROVED` as the only signal to
    proceed past the gate; absence or `REJECTED` keeps the job parked
    (or marks it failed on the rejection path).
    """

    model_config = ConfigDict(extra="forbid")

    decision: ApprovalDecision
    granted_at: datetime
    granted_by: str | None = None
    dag_edits: DagEdit | None = None
    appended_context: str | None = Field(default=None, max_length=4000)
    reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _rejection_requires_reason(self) -> "HumanApproval":
        if self.decision == ApprovalDecision.REJECTED and not (self.reason and self.reason.strip()):
            raise ValueError("reason is required when decision is REJECTED")
        return self

    @model_validator(mode="after")
    def _granted_at_is_tz_aware(self) -> "HumanApproval":
        if self.granted_at.tzinfo is None:
            raise ValueError("granted_at must be timezone-aware")
        return self

    @classmethod
    def approve(
        cls,
        *,
        granted_by: str | None = None,
        dag_edits: DagEdit | None = None,
        appended_context: str | None = None,
    ) -> "HumanApproval":
        """Construct an APPROVED decision with `granted_at = now (UTC)`."""
        return cls(
            decision=ApprovalDecision.APPROVED,
            granted_at=datetime.now(timezone.utc),
            granted_by=granted_by,
            dag_edits=dag_edits,
            appended_context=appended_context,
        )

    @classmethod
    def reject(cls, *, reason: str, granted_by: str | None = None) -> "HumanApproval":
        """Construct a REJECTED decision with `granted_at = now (UTC)`."""
        return cls(
            decision=ApprovalDecision.REJECTED,
            granted_at=datetime.now(timezone.utc),
            granted_by=granted_by,
            reason=reason,
        )
