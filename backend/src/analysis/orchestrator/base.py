"""Orchestrator protocol + shared substrate for orchestrator implementations.

The orchestrator coordinates specialist agents to run a causal-inference
analysis end to end. JobManager depends on this Protocol rather than a
concrete class, so swapping between standard and react orchestration
is a configuration choice, not a code change.

This module also hosts cross-orchestrator helpers — anything both the
standard and react implementations must agree on. The first such helper
is the human-approval gate (post-profile, pre-analysis): the orchestrator
checks `should_pause_for_approval` once the data is profiled and parks
identically via `park_for_approval`, so the human reviews the downloaded
data before any analysis runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable, Protocol, runtime_checkable

from src.analysis.agents.base.state import AnalysisState, JobStatus
from src.domain.approval import ApprovalDecision

if TYPE_CHECKING:
    from src.analysis.agents.base.agent import BaseAgent


@runtime_checkable
class Orchestrator(Protocol):
    """Contract every orchestrator implementation must satisfy.

    Implementations today:
        StandardOrchestrator (standard/agent.py): LLM-driven dispatch
            with a default workflow suggested in the system prompt.
        ReActOrchestrator (react/agent.py): Fully autonomous ReAct loop
            with no fixed workflow.

    Both inherit execute_with_tracing from BaseAgent / ReActAgent, which
    is the shared tracing and logging wrapper. JobManager invokes that
    wrapper rather than execute directly so AgentTrace recording and
    structlog emission happen uniformly across orchestrators.
    """

    def register_specialist(self, name: str, agent: BaseAgent) -> None:
        """Register a specialist the orchestrator can dispatch to."""
        ...

    async def execute_with_tracing(self, state: AnalysisState) -> AnalysisState:
        """Run the orchestration loop with tracing and logging applied.

        Returns the updated state. In-pipeline failures mark the state
        FAILED rather than raising. May also return with
        `state.status == AWAITING_APPROVAL` when the human-approval gate
        fires — that is a *yield*, not a terminus, and the worker layer
        handles persistence + later resumption.
        """
        ...


# ── Human-approval gate (shared substrate) ─────────────────────────────────


StatusCallback = Callable[[AnalysisState], Awaitable[None]]


def should_pause_for_approval(state: AnalysisState) -> bool:
    """Truth-table check for the data-review gate (post-profile, pre-analysis).

    Returns True iff the orchestrator should park the job so the human can
    review the downloaded data before any analysis runs. The gate fires
    exactly once per job:

    - if a prior APPROVED decision is on state, we have already passed
      the gate — do not pause again;
    - if the data has not been profiled yet, there is nothing to review
      — do not pause;
    - if EDA has started or any effect estimate exists, we are past the
      data stage (a resume case re-entering the loop) — do not pause;
    - otherwise the gate fires.
    """
    approval = state.human_approval
    if approval is not None and approval.decision == ApprovalDecision.APPROVED:
        return False
    if state.data_profile is None:
        return False
    if state.eda_result is not None or state.treatment_effects:
        return False
    return True


def _build_gate_payload(state: AnalysisState) -> dict:
    """Compact snapshot the SSE event and approval endpoint carry at the gate.

    Pulled from slots the data_profiler has already populated: the dataset
    shape, the downloaded file list, and the treatment/outcome candidates.
    The frontend's primary review surface is the F1 dataset view; this is
    the headline summary alongside it.
    """
    profile = state.data_profile

    data_summary: dict | None = None
    if profile is not None:
        data_summary = {
            "n_samples": profile.n_samples,
            "n_features": profile.n_features,
            "treatment_candidates": list(profile.treatment_candidates)[:8],
            "outcome_candidates": list(profile.outcome_candidates)[:8],
        }

    files = [
        {"name": f.name, "format": f.format, "used": f.used}
        for f in state.dataset_info.files
    ]

    return {
        "treatment_variable": state.treatment_variable,
        "outcome_variable": state.outcome_variable,
        "data_summary": data_summary,
        "files": files,
    }


async def park_for_approval(
    state: AnalysisState,
    status_callback: StatusCallback | None = None,
) -> AnalysisState:
    """Set AWAITING_APPROVAL, emit the SSE gate event, persist via callback.

    The orchestrator returns the state to its caller right after this so
    the asyncio.Task ends cleanly. The worker layer (`_run_job_inner`)
    sees the parked status, writes the full state to the parked-states
    store, and exits without saving results. The job then waits for the
    approval API to load the parked state and respawn the task.
    """
    state.status = JobStatus.AWAITING_APPROVAL
    payload = _build_gate_payload(state)
    state.push_sse_event("approval_required", payload)
    if status_callback is not None:
        await status_callback(state)
    return state
