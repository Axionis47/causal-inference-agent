"""run_discovery_algorithm - dispatches PC / GES / NOTEARS / LiNGAM with empty-DAG retry."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from ..helpers import create_simple_dag, run_algorithm, select_columns

SCHEMA = {
    "name": "run_discovery_algorithm",
    "description": "Run a causal discovery algorithm on the data. Choose based on data characteristics.",
    "parameters": {
        "type": "object",
        "properties": {
            "algorithm": {
                "type": "string",
                "enum": ["pc", "ges", "notears", "lingam"],
                "description": "Discovery algorithm to run",
            },
            "alpha": {
                "type": "number",
                "description": "Significance level for independence tests (default: 0.05)",
            },
            "threshold": {
                "type": "number",
                "description": "Edge weight threshold for weighted methods (default: 0.1)",
            },
        },
        "required": ["algorithm"],
    },
}

# Order to try when the chosen algorithm produces a sparse graph.
_RETRY_ORDER = {"pc": "ges", "ges": "notears", "notears": None, "lingam": "ges"}


async def handle(
    agent,
    state: AnalysisState,
    algorithm: str = "pc",
    alpha: float = 0.05,
    threshold: float = 0.1,
    **kwargs,
) -> ToolResult:
    if agent._df is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Dataset not loaded",
        )

    df_work, relevant_cols = select_columns(state, agent._df, agent._treatment_var, agent._outcome_var)
    df_subset = df_work[relevant_cols].dropna()

    if len(df_subset) < 50:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=f"Not enough complete cases ({len(df_subset)}) for discovery",
        )

    try:
        dag = run_algorithm(
            df_subset, algorithm, alpha, threshold,
            agent._treatment_var, agent._outcome_var, agent.logger,
        )

        if dag:
            if agent._treatment_var:
                dag.treatment_variable = agent._treatment_var
            if agent._outcome_var:
                dag.outcome_variable = agent._outcome_var

            agent._discovered_graphs[algorithm] = dag
            agent._current_graph = dag

            n_edges = len(dag.edges)
            n_nodes = len(dag.nodes)
            n_samples = len(df_subset)

            if n_edges < 2:
                next_algo = _RETRY_ORDER.get(algorithm)
                if next_algo and next_algo not in agent._discovered_graphs:
                    state.push_decision(
                        agent="causal_discovery",
                        decision_type="algorithm_retry",
                        choice=f"retrying with {next_algo}",
                        reason=f"{algorithm} produced only {n_edges} edges on {n_samples} samples - insufficient for causal structure",
                    )
                    agent.logger.warning(
                        "discovery_sparse_dag_retry",
                        algorithm=algorithm, n_edges=n_edges, next_algo=next_algo,
                    )
                    return await handle(agent, state, algorithm=next_algo, alpha=alpha, threshold=threshold)

                agent.logger.warning(
                    "discovery_all_retries_exhausted",
                    algorithm=algorithm, n_edges=n_edges,
                )
                profile = agent._current_state.data_profile if agent._current_state else None
                fallback_dag = create_simple_dag(agent._treatment_var, agent._outcome_var, profile)
                agent._discovered_graphs["fallback"] = fallback_dag
                agent._current_graph = fallback_dag

                state.push_decision(
                    agent="causal_discovery",
                    decision_type="algorithm_retry",
                    choice="fallback to minimal DAG",
                    reason="All algorithms produced fewer than 2 edges - using minimal treatment->outcome DAG",
                )
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={
                        "algorithm": "FALLBACK",
                        "fallback": True,
                        "message": "All algorithms produced sparse graphs. Using minimal treatment->outcome DAG.",
                        "n_nodes": len(fallback_dag.nodes),
                        "n_edges": len(fallback_dag.edges),
                        "variables_used": len(relevant_cols),
                        "samples_used": n_samples,
                    },
                )

            directed = [e for e in dag.edges if e.edge_type == "directed"]
            undirected = [e for e in dag.edges if e.edge_type == "undirected"]

            has_direct_edge = False
            if agent._treatment_var and agent._outcome_var:
                has_direct_edge = any(
                    e.source == agent._treatment_var and e.target == agent._outcome_var
                    for e in directed
                )

            state.push_decision(
                agent="causal_discovery",
                decision_type="algorithm_succeeded",
                choice=algorithm.upper(),
                reason=f"{algorithm.upper()} produced {len(directed)} directed and {len(undirected)} undirected edges across {n_nodes} nodes using {n_samples} samples",
            )

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "algorithm": algorithm.upper(),
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "n_directed": len(directed),
                    "n_undirected": len(undirected),
                    "variables_used": len(relevant_cols),
                    "samples_used": n_samples,
                    "treatment_outcome_edge": has_direct_edge,
                    "message": "Discovery completed. Call inspect_graph for details.",
                },
            )

        # Engine + legacy both returned None -> fallback DAG.
        profile = agent._current_state.data_profile if agent._current_state else None
        fallback = create_simple_dag(agent._treatment_var, agent._outcome_var, profile)
        agent._discovered_graphs[algorithm] = fallback
        agent._current_graph = fallback

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "algorithm": algorithm.upper(),
                "fallback": True,
                "message": "Algorithm failed - using simple DAG fallback",
                "n_nodes": len(fallback.nodes),
                "n_edges": len(fallback.edges),
            },
        )

    except Exception as e:
        agent.logger.error("discovery_algorithm_error", algorithm=algorithm, error=str(e))
        profile = agent._current_state.data_profile if agent._current_state else None
        fallback = create_simple_dag(agent._treatment_var, agent._outcome_var, profile)
        agent._discovered_graphs[algorithm] = fallback
        agent._current_graph = fallback

        state.push_decision(
            agent="causal_discovery",
            decision_type="algorithm_failed",
            choice=f"{algorithm.upper()} (fallback to simple DAG)",
            reason=f"{algorithm.upper()} failed with error: {str(e)[:200]}; falling back to simple treatment-outcome DAG",
        )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "algorithm": algorithm.upper(),
                "error": str(e),
                "fallback": True,
                "message": f"Error during discovery: {str(e)}. Using simple DAG fallback.",
                "n_nodes": len(fallback.nodes),
                "n_edges": len(fallback.edges),
            },
        )
