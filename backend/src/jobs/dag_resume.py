"""Resume routing for the DAG gate (checkpoint A).

A parked job at the DAG checkpoint (should_pause_for_dag_approval) is routed here.
Three actions:

  APPROVED: record dag_approval, apply any structured DAG edits, append context,
            respawn past the gate into estimation.
  REVISE:   append the note, clear refined_dag so dag_expert re-runs with it,
            respawn; the gate fires again on the redone DAG. Bounded by
            MAX_GATE_REVISIONS so the redo loop cannot run forever.
  REJECTED: record dag_approval, mark the job FAILED with the reason.

Generic plumbing (append_context, respawn, RevisionLimitReached) lives in
gate_resume.py and is shared with the results gate.
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

# Kept for callers/tests that import the DAG-specific name.
MAX_DAG_REVISIONS = MAX_GATE_REVISIONS

__all__ = ["MAX_DAG_REVISIONS", "RevisionLimitReached", "resume_dag_gate"]


def _apply_dag_edits(state: AnalysisState, approval: HumanApproval) -> None:
    """Merge the human's structured edits into the refined DAG in place."""
    target = state.refined_dag or state.discovered_dag
    if target is None or approval.dag_edits is None:
        return
    edits = approval.dag_edits
    if edits.adjustment_set is not None:
        target.adjustment_set = list(edits.adjustment_set)
    if edits.forbidden_edges is not None:
        target.forbidden_edges = list(edits.forbidden_edges)
    if edits.variable_roles is not None:
        target.variable_roles = dict(edits.variable_roles)


async def resume_dag_gate(
    manager: "JobManager",
    job_id: str,
    state: AnalysisState,
    approval: HumanApproval,
) -> dict[str, Any]:
    """Apply a human decision at the DAG gate and resume, redo, or fail."""
    if approval.decision == ApprovalDecision.APPROVED:
        _apply_dag_edits(state, approval)
        append_context(state, approval)
        state.dag_approval = approval
        state.status = JobStatus.DISCOVERING_CAUSAL
        state.push_sse_event(
            "dag_approval_granted",
            {
                "decision": approval.decision.value,
                "appended_context_chars": len(approval.appended_context or ""),
            },
        )
        await respawn(manager, state)
        return {"resumed": True, "status": state.status.value}

    if approval.decision == ApprovalDecision.REVISE:
        if state.dag_revision_count >= MAX_GATE_REVISIONS:
            raise RevisionLimitReached(
                f"maximum DAG revisions ({MAX_GATE_REVISIONS}) reached; "
                "approve or reject the DAG"
            )
        state.dag_revision_count += 1
        append_context(state, approval)
        # Clear the refined DAG so dag_expert re-runs with the new note; keep
        # dag_approval None so the gate fires again on the redone DAG.
        state.refined_dag = None
        state.status = JobStatus.DISCOVERING_CAUSAL
        state.push_sse_event(
            "dag_revision_requested",
            {
                "revision": state.dag_revision_count,
                "appended_context_chars": len(approval.appended_context or ""),
            },
        )
        await respawn(manager, state)
        return {"resumed": True, "status": state.status.value}

    # REJECTED
    state.dag_approval = approval
    state.mark_failed(f"Rejected at DAG gate: {approval.reason}", "human_approval")
    await manager.firestore.update_job(state)
    await manager.firestore.save_traces(state)
    await manager.firestore.delete_parked_state(job_id)
    return {"resumed": False, "status": state.status.value}
