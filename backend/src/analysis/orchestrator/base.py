"""Orchestrator protocol + shared substrate for orchestrator implementations.

The orchestrator coordinates specialist agents to run a causal-inference
analysis end to end. JobManager depends on this Protocol rather than a
concrete class, so swapping between standard and react orchestration
is a configuration choice, not a code change.

This module also hosts cross-orchestrator helpers — anything both the
standard and react implementations must agree on. The first such helper
is the human-approval gate (post-DAG, pre-estimation): both orchestrators
must check `should_pause_for_approval` at the same boundary and park
identically via `park_for_approval`, so the contract cannot drift.
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
    """Truth-table check for the post-DAG / pre-estimation gate.

    Returns True iff the orchestrator should park the job and wait for a
    human review. The gate fires exactly once per job:

    - if a prior APPROVED decision is on state, we have already passed
      the gate — do not pause again;
    - if no DAG has been built yet (neither refined nor discovered),
      there is nothing to review — do not pause;
    - if the EDA summary has not landed yet, the review snapshot would
      be missing half its content — do not pause;
    - if effect estimation has produced any results, we are past the
      gate (a resume case re-entering the loop) — do not pause;
    - otherwise the gate fires.
    """
    approval = state.human_approval
    if approval is not None and approval.decision == ApprovalDecision.APPROVED:
        return False
    if state.refined_dag is None and state.discovered_dag is None:
        return False
    if state.eda_result is None:
        return False
    if state.treatment_effects:
        return False
    return True


def _build_gate_payload(state: AnalysisState) -> dict:
    """Compact snapshot the UI renders at the approval gate.

    Pulled from state slots the sealed agents have already populated:
    EDA findings, the proposed DAG (refined preferred), and sealed-agent
    briefs (flags + headlines). Method selection is deliberately absent
    — the effect_estimator picks methods at runtime from observed
    conditions, so showing a precommitted list would mislead.
    """
    eda = state.eda_result
    dag = state.refined_dag if state.refined_dag is not None else state.discovered_dag

    eda_summary: dict | None = None
    if eda is not None:
        eda_summary = {
            "data_quality_score": eda.data_quality_score,
            "data_quality_issues": list(eda.data_quality_issues)[:5],
            "balance_summary": eda.balance_summary,
            "high_correlations": list(eda.high_correlations)[:5],
        }

    dag_snapshot: dict | None = None
    if dag is not None:
        dag_snapshot = {
            "nodes": list(dag.nodes),
            "edges": [e.model_dump() for e in dag.edges],
            "discovery_method": dag.discovery_method,
            "interpretation": dag.interpretation,
            "adjustment_set": list(dag.adjustment_set) if dag.adjustment_set else None,
            "variable_roles": dict(dag.variable_roles) if dag.variable_roles else None,
            "forbidden_edges": list(dag.forbidden_edges) if dag.forbidden_edges else None,
        }

    brief_flags: dict[str, dict] = {}
    for name in ("eda_agent", "causal_discovery", "domain_knowledge", "dag_expert"):
        brief = state.agent_briefs.get(name)
        if brief is not None:
            brief_flags[name] = {
                "status": brief.status,
                "headline": brief.headline,
                "flags": [f.value for f in brief.flags],
                "raised_issues": list(brief.raised_issues)[:3],
            }

    return {
        "treatment_variable": state.treatment_variable,
        "outcome_variable": state.outcome_variable,
        "eda_summary": eda_summary,
        "proposed_dag": dag_snapshot,
        "brief_flags": brief_flags,
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
