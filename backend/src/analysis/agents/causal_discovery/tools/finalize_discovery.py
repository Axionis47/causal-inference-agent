"""finalize_discovery - record the chosen algorithm and flip _finalized."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from ..helpers import assess_dag_quality

SCHEMA = {
    "name": "finalize_discovery",
    "description": "Finalize the causal discovery with chosen graph. Call this when satisfied with the discovered structure.",
    "parameters": {
        "type": "object",
        "properties": {
            "chosen_algorithm": {
                "type": "string",
                "description": "Which algorithm's result to use",
            },
            "confounders": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Identified confounder variables",
            },
            "mediators": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Identified mediator variables",
            },
            "interpretation": {
                "type": "string",
                "description": "Overall interpretation of the discovered structure",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Confidence in the discovered structure",
            },
        },
        "required": ["chosen_algorithm", "interpretation", "confidence"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    chosen_algorithm: str = "",
    interpretation: str = "",
    confidence: str = "medium",
    confounders: list[str] | None = None,
    mediators: list[str] | None = None,
    **kwargs,
) -> ToolResult:
    agent.logger.info(
        "discovery_finalizing",
        chosen_algorithm=chosen_algorithm,
        confidence=confidence,
    )

    n_algorithms_run = len(agent._discovered_graphs)
    chosen_dag = agent._discovered_graphs.get(chosen_algorithm)
    n_edges = len(chosen_dag.edges) if chosen_dag else 0
    n_nodes = len(chosen_dag.nodes) if chosen_dag else 0

    quality = assess_dag_quality(n_edges, n_nodes)

    state.push_decision(
        agent="causal_discovery",
        decision_type="dag_quality_assessment",
        choice=quality,
        reason=f"DAG has {n_edges} edges across {n_nodes} nodes - quality: {quality}",
    )

    final_interpretation = interpretation
    if quality in ("poor", "empty"):
        final_interpretation = interpretation + (
            " Warning: The discovered graph has very few edges, suggesting "
            "the algorithm may lack power for this dataset. Downstream "
            "confounder selection will rely on statistical methods."
        )

    n_samples = len(agent._df) if agent._df is not None else 0
    if n_algorithms_run == 1 and n_samples > 100:
        state.push_decision(
            agent="causal_discovery",
            decision_type="single_algorithm_warning",
            choice=chosen_algorithm,
            reason=f"Only 1 algorithm was run on {n_samples} samples - consider running a second algorithm for robustness",
        )
        agent.logger.warning(
            "discovery_single_algorithm",
            chosen_algorithm=chosen_algorithm,
            n_samples=n_samples,
        )

    agent._final_result = {
        "chosen_algorithm": chosen_algorithm,
        "confounders": confounders or [],
        "mediators": mediators or [],
        "interpretation": final_interpretation,
        "confidence": confidence,
    }
    agent._finalized = True

    state.push_decision(
        agent="causal_discovery",
        decision_type="final_dag_selected",
        choice=chosen_algorithm,
        reason=f"Selected {chosen_algorithm.upper()} (confidence: {confidence}) from {n_algorithms_run} algorithm(s) run; graph has {n_edges} edges, {len(confounders or [])} confounders identified; quality: {quality}",
    )

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "discovery_finalized": True,
            "chosen_algorithm": chosen_algorithm,
            "confidence": confidence,
            "n_confounders": len(confounders or []),
            "n_mediators": len(mediators or []),
            "dag_quality": quality,
            "n_algorithms_run": n_algorithms_run,
        },
    )
