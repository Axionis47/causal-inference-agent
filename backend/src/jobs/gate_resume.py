"""Shared plumbing for the human-gate resumes (DAG and results gates).

Both gates resume the same way: apply the human decision, then respawn the run
or fail it. The gate-specific bits (which field holds the approval, what a REVISE
clears so the right stage re-runs) live in dag_resume.py and results_resume.py;
the generic plumbing is here so the two stay in step.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.analysis.agents.base.state import AnalysisState
from src.domain.approval import HumanApproval

if TYPE_CHECKING:
    from src.jobs.manager import JobManager

# Bound on REVISE rounds at any human gate, so a redo loop cannot run forever.
MAX_GATE_REVISIONS = 3


class RevisionLimitReached(ValueError):
    """A REVISE that would exceed MAX_GATE_REVISIONS. Maps to HTTP 409."""


def append_context(state: AnalysisState, approval: HumanApproval) -> None:
    """Append the human's note to the analyst context (existing prose first)."""
    if approval.appended_context:
        existing = state.dataset_info.user_provided_context or ""
        state.dataset_info.user_provided_context = (
            existing + "\n\n" + approval.appended_context
        ).strip()


async def respawn(manager: "JobManager", state: AnalysisState) -> None:
    """Delete the parked record, persist the job, spawn a fresh run task."""
    await manager.firestore.delete_parked_state(state.job_id)
    await manager.firestore.update_job(state)
    task = asyncio.create_task(manager._run_job(state))
    async with manager._jobs_lock:
        manager._running_jobs[state.job_id] = task
