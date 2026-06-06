"""get_adjustment_set - backdoor adjustment set over the refined DAG."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from ..helpers import adjustment_set_from_dag

SCHEMA = {
    "name": "get_adjustment_set",
    "description": (
        "Get the proper adjustment set for estimating the causal effect "
        "based on the validated DAG using backdoor criterion."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "treatment": {"type": "string"},
            "outcome": {"type": "string"},
        },
        "required": ["treatment", "outcome"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    treatment: str = "",
    outcome: str = "",
    **kwargs,
) -> ToolResult:
    if not state.refined_dag:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="No DAG available. Run fuse_and_validate first.",
        )

    result = adjustment_set_from_dag(state.refined_dag, treatment, outcome)
    adjustment_set = result["adjustment_set"]
    confounders = result["confounders"]
    mediators = result["mediators"]
    colliders = result["colliders"]

    state.refined_dag.adjustment_set = adjustment_set

    state.push_decision(
        agent="dag_expert",
        decision_type="adjustment_set_selected",
        choice=", ".join(adjustment_set) if adjustment_set else "(empty set)",
        reason=(
            f"Backdoor criterion identifies {len(adjustment_set)} variable(s) to "
            f"block all non-causal paths from {treatment} to {outcome}; "
            f"excluded {len(mediators)} mediator(s) and {len(colliders)} collider(s)"
        ),
    )

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "treatment": treatment,
            "outcome": outcome,
            "adjustment_set": adjustment_set,
            "confounders": confounders,
            "mediators": mediators,
            "colliders_do_not_adjust": colliders,
            "recommendation": (
                f"Adjust for {adjustment_set} to block backdoor paths. "
                f"Do NOT adjust for {mediators} (mediators) or "
                f"{colliders} (colliders)."
            ),
        },
    )
