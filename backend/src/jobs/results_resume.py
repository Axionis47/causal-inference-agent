"""Resume routing for the results gate (checkpoint B).

A parked job at the results checkpoint (should_pause_for_results) is routed here.
Three actions:

  APPROVED: record results_approval, proceed into critique and finalize.
  REVISE:   append the note, clear treatment_effects and sensitivity_results so
            the estimation tail re-runs (and now sees the note, via the base
            agent's note threading), respawn; the gate fires again on the fresh
            estimates. Bounded by MAX_GATE_REVISIONS.
  REJECTED: record results_approval, mark the job FAILED with the reason.

Generic plumbing is shared with the DAG gate (gate_resume.py).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.analysis.agents.base.state import AnalysisState, JobStatus
from src.domain.approval import ApprovalDecision, HumanApproval
from src.jobs.gate_resume import (
    MAX_GATE_REVISIONS,
    RevisionLimitReached,
    append_context,
    respawn,
)

if TYPE_CHECKING:
    from src.jobs.manager import JobManager

__all__ = ["resume_results_gate"]


async def resume_results_gate(
    manager: "JobManager",
    job_id: str,
    state: AnalysisState,
    approval: HumanApproval,
) -> dict[str, Any]:
    """Apply a human decision at the results gate and finalize, re-estimate, or fail."""
    if approval.decision == ApprovalDecision.APPROVED:
        append_context(state, approval)
        state.results_approval = approval
        state.status = JobStatus.CRITIQUE_REVIEW
        state.push_sse_event(
            "results_approval_granted",
            {
                "decision": approval.decision.value,
                "appended_context_chars": len(approval.appended_context or ""),
            },
        )
        await respawn(manager, state)
        return {"resumed": True, "status": state.status.value}

    if approval.decision == ApprovalDecision.REVISE:
        if state.results_revision_count >= MAX_GATE_REVISIONS:
            raise RevisionLimitReached(
                f"maximum results revisions ({MAX_GATE_REVISIONS}) reached; "
                "approve or reject the results"
            )
        state.results_revision_count += 1
        append_context(state, approval)
        # Clear the estimates so the estimation tail (effect_estimator,
        # ps_diagnostics, sensitivity) re-runs with the note; keep
        # results_approval None so the gate fires again on the fresh estimates.
        state.treatment_effects = []
        state.sensitivity_results = []
        state.status = JobStatus.ESTIMATING_EFFECTS
        state.push_sse_event(
            "results_revision_requested",
            {
                "revision": state.results_revision_count,
                "appended_context_chars": len(approval.appended_context or ""),
            },
        )
        await respawn(manager, state)
        return {"resumed": True, "status": state.status.value}

    # REJECTED
    state.results_approval = approval
    state.mark_failed(f"Rejected at results gate: {approval.reason}", "human_approval")
    await manager.firestore.update_job(state)
    await manager.firestore.save_traces(state)
    await manager.firestore.delete_parked_state(job_id)
    return {"resumed": False, "status": state.status.value}
