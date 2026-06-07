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


# ── DAG gate (checkpoint A: post-dag_expert, pre-estimation) ────────────────


def should_pause_for_dag_approval(state: AnalysisState) -> bool:
    """Truth-table check for the DAG gate (post-dag_expert, pre-estimation).

    The DAG gate always comes after the data gate, so it uses `dag_approval`
    (separate from `human_approval`) and also requires the data to have been
    approved. That second condition keeps resume routing unambiguous: a state
    with `human_approval` unset is at the data gate, not here. Returns True iff:

    - the DAG has not already been APPROVED;
    - the data gate has been passed (human_approval APPROVED);
    - dag_expert has produced a refined DAG;
    - estimation has not started (no effects).

    On a REVISE, refined_dag is cleared and dag_approval stays None, so a freshly
    refined DAG re-fires the gate.
    """
    approval = state.dag_approval
    if approval is not None and approval.decision == ApprovalDecision.APPROVED:
        return False
    data = state.human_approval
    if data is None or data.decision != ApprovalDecision.APPROVED:
        return False
    if state.refined_dag is None:
        return False
    if state.treatment_effects:
        return False
    return True


def _dag_justification(state: AnalysisState, dag) -> dict:
    """The "why this DAG, and is the effect identified" block for the gate.

    Assembled deterministically from the DAG and dag_expert's sealed brief, so a
    human can disagree on concrete grounds: the identification statement (does an
    adjustment set identify the effect, or not), the adjustment set, the agent's
    own flags (no_adjustment_set, collider_suspected, dag_conflict), the concerns
    it raised, and its narrative interpretation of the graph.
    """
    brief = state.agent_briefs.get("dag_expert")
    flags = [f.value for f in brief.flags] if brief else []
    concerns = list(brief.raised_issues) if brief else []

    adjustment_set = list(dag.adjustment_set or [])
    t, y = state.treatment_variable, state.outcome_variable
    if not adjustment_set or "no_adjustment_set" in flags:
        identification = (
            f"No backdoor adjustment set was found, so the effect of {t} on {y} "
            "is not identified from this DAG alone; estimation falls back to the "
            "profiler's candidate confounders."
        )
    else:
        identification = (
            f"The effect of {t} on {y} is identified by adjusting for "
            f"{', '.join(adjustment_set)}."
        )

    return {
        "identification": identification,
        "adjustment_set": adjustment_set,
        "interpretation": dag.interpretation,
        "flags": flags,
        "concerns": concerns,
    }


def _build_dag_gate_payload(state: AnalysisState) -> dict:
    """Snapshot the SSE event and approval endpoint carry at the DAG gate.

    The refined DAG (or the discovery fallback) as plain dicts the frontend can
    render (nodes, edges, adjustment set, variable roles, forbidden edges), plus
    a justification block (identification + the agent's flags and concerns) so
    the review is on concrete grounds. The interactive figure is rendered on the
    frontend from these nodes and edges.
    """
    dag = state.refined_dag or state.discovered_dag

    dag_summary: dict | None = None
    justification: dict | None = None
    if dag is not None:
        dag_summary = {
            "nodes": list(dag.nodes),
            "edges": [
                {"source": e.source, "target": e.target, "edge_type": e.edge_type}
                for e in dag.edges
            ],
            "adjustment_set": list(dag.adjustment_set or []),
            "variable_roles": dict(dag.variable_roles or {}),
            "forbidden_edges": list(dag.forbidden_edges or []),
            "discovery_method": dag.discovery_method,
            "interpretation": dag.interpretation,
        }
        justification = _dag_justification(state, dag)

    return {
        "gate": "dag",
        "treatment_variable": state.treatment_variable,
        "outcome_variable": state.outcome_variable,
        "dag": dag_summary,
        "justification": justification,
    }


async def park_for_dag_approval(
    state: AnalysisState,
    status_callback: StatusCallback | None = None,
) -> AnalysisState:
    """Park at the DAG gate: set AWAITING_APPROVAL, emit the DAG SSE event.

    Mirrors park_for_approval but emits `dag_approval_required` (the data gate's
    `approval_required` is left untouched) so the frontend can tell the two
    gates apart while they share the AWAITING_APPROVAL status. Resume routing
    reads the parked state to decide which gate the decision applies to.
    """
    state.status = JobStatus.AWAITING_APPROVAL
    payload = _build_dag_gate_payload(state)
    state.push_sse_event("dag_approval_required", payload)
    if status_callback is not None:
        await status_callback(state)
    return state


# ── Results gate (checkpoint B: post-sensitivity, pre-critique) ─────────────


def should_pause_for_results(state: AnalysisState) -> bool:
    """Truth-table check for the results gate (post-sensitivity, pre-critique).

    Fires once the estimates and their robustness checks exist, for a human to
    review before the analysis is finalized. Uses `results_approval`, separate
    from the earlier gates, and requires the DAG gate already passed so resume
    routing stays unambiguous. Returns True iff:

    - the results have not already been APPROVED;
    - the DAG gate has been passed (dag_approval APPROVED);
    - effect estimates exist;
    - sensitivity results exist.

    On a REVISE, treatment_effects and sensitivity_results are cleared so the
    estimation tail re-runs, and results_approval stays None, so fresh estimates
    re-fire the gate.
    """
    approval = state.results_approval
    if approval is not None and approval.decision == ApprovalDecision.APPROVED:
        return False
    dag = state.dag_approval
    if dag is None or dag.decision != ApprovalDecision.APPROVED:
        return False
    if not state.treatment_effects:
        return False
    if not state.sensitivity_results:
        return False
    return True


def _build_results_gate_payload(state: AnalysisState) -> dict:
    """Snapshot the SSE event and approval endpoint carry at the results gate.

    The estimates (a forest plot of estimate + CI per method), the sensitivity
    verdicts, the propensity-score diagnostics, and per-covariate balance (the
    Love plot). The frontend renders the plots from these numbers.
    """
    effects = [
        {
            "method": e.method,
            "estimand": e.estimand,
            "estimate": e.estimate,
            "ci_lower": e.ci_lower,
            "ci_upper": e.ci_upper,
            "p_value": e.p_value,
        }
        for e in state.treatment_effects
    ]
    sensitivity = [
        {
            "method": s.method,
            "robustness_value": s.robustness_value,
            "interpretation": s.interpretation,
        }
        for s in state.sensitivity_results
    ]
    balance: list[dict] = []
    eda = state.eda_result
    if eda is not None and eda.covariate_balance:
        balance = [
            {"covariate": cov, "smd": metrics.get("smd")}
            for cov, metrics in eda.covariate_balance.items()
            if isinstance(metrics, dict) and metrics.get("smd") is not None
        ]

    return {
        "gate": "results",
        "treatment_variable": state.treatment_variable,
        "outcome_variable": state.outcome_variable,
        "effects": effects,
        "sensitivity": sensitivity,
        "ps_diagnostics": state.ps_diagnostics,
        "balance": balance,
    }


async def park_for_results_approval(
    state: AnalysisState,
    status_callback: StatusCallback | None = None,
) -> AnalysisState:
    """Park at the results gate: set AWAITING_APPROVAL, emit the results event."""
    state.status = JobStatus.AWAITING_APPROVAL
    payload = _build_results_gate_payload(state)
    state.push_sse_event("results_approval_required", payload)
    if status_callback is not None:
        await status_callback(state)
    return state
