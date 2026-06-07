"""Resume routing for the DAG gate (checkpoint A).

The data gate's resume lives on JobManager.resume_from_approval. A parked job at
the DAG checkpoint (should_pause_for_dag_approval) is routed here instead. Three
actions:

  APPROVED: record dag_approval, apply any structured DAG edits, append context,
            respawn past the gate into estimation.
  REVISE:   append the note, clear refined_dag so dag_expert re-runs with it,
            respawn; the gate fires again on the redone DAG. Bounded by
            MAX_DAG_REVISIONS so the redo loop cannot run forever.
  REJECTED: record dag_approval, mark the job FAILED with the reason.

This lives outside manager.py (already over the size cap) but reaches into the
manager's storage + task plumbing, matching the data-gate resume it mirrors.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.analysis.agents.base.state import AnalysisState, JobStatus
from src.domain.approval import ApprovalDecision, HumanApproval

if TYPE_CHECKING:
    from src.jobs.manager import JobManager

MAX_DAG_REVISIONS = 3


class RevisionLimitReached(ValueError):
    """A REVISE that would exceed MAX_DAG_REVISIONS. Maps to HTTP 409."""


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


def _append_context(state: AnalysisState, approval: HumanApproval) -> None:
    """Append the human's note to the analyst context (existing prose first)."""
    if approval.appended_context:
        existing = state.dataset_info.user_provided_context or ""
        state.dataset_info.user_provided_context = (
            existing + "\n\n" + approval.appended_context
        ).strip()


async def _respawn(manager: "JobManager", state: AnalysisState) -> None:
    """Delete the parked record, persist the job, spawn a fresh run task."""
    await manager.firestore.delete_parked_state(state.job_id)
    await manager.firestore.update_job(state)
    task = asyncio.create_task(manager._run_job(state))
    async with manager._jobs_lock:
        manager._running_jobs[state.job_id] = task


async def resume_dag_gate(
    manager: "JobManager",
    job_id: str,
    state: AnalysisState,
    approval: HumanApproval,
) -> dict[str, Any]:
    """Apply a human decision at the DAG gate and resume, redo, or fail."""
    if approval.decision == ApprovalDecision.APPROVED:
        _apply_dag_edits(state, approval)
        _append_context(state, approval)
        state.dag_approval = approval
        state.status = JobStatus.DISCOVERING_CAUSAL
        state.push_sse_event(
            "dag_approval_granted",
            {
                "decision": approval.decision.value,
                "appended_context_chars": len(approval.appended_context or ""),
            },
        )
        await _respawn(manager, state)
        return {"resumed": True, "status": state.status.value}

    if approval.decision == ApprovalDecision.REVISE:
        if state.dag_revision_count >= MAX_DAG_REVISIONS:
            raise RevisionLimitReached(
                f"maximum DAG revisions ({MAX_DAG_REVISIONS}) reached; "
                "approve or reject the DAG"
            )
        state.dag_revision_count += 1
        _append_context(state, approval)
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
        await _respawn(manager, state)
        return {"resumed": True, "status": state.status.value}

    # REJECTED
    state.dag_approval = approval
    state.mark_failed(f"Rejected at DAG gate: {approval.reason}", "human_approval")
    await manager.firestore.update_job(state)
    await manager.firestore.save_traces(state)
    await manager.firestore.delete_parked_state(job_id)
    return {"resumed": False, "status": state.status.value}
