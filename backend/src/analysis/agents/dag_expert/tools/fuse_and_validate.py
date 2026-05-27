"""fuse_and_validate - merge domain and data edges, resolve conflicts, write refined_dag."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from ..helpers import fuse_edges

SCHEMA = {
    "name": "fuse_and_validate",
    "description": (
        "Fuse domain edges with discovery edges to create final DAG. "
        "Resolves conflicts, applies forbidden edge constraints, "
        "and assigns final confidence scores."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conflict_resolution": {
                "type": "string",
                "enum": ["domain_priority", "data_priority", "consensus_only"],
                "description": "How to resolve conflicts between domain and data",
            },
        },
        "required": ["conflict_resolution"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    conflict_resolution: str = "domain_priority",
    **kwargs,
) -> ToolResult:
    validated_dag, conflicts, edge_sources = fuse_edges(
        domain_edges=agent._domain_edges,
        data_edges=agent._data_edges,
        forbidden_edges=agent._forbidden_edges,
        conflict_resolution=conflict_resolution,
        treatment_var=state.treatment_variable,
        outcome_var=state.outcome_variable,
        variable_roles=agent._variable_roles,
    )

    state.refined_dag = validated_dag

    n_domain = len([e for e in edge_sources.values() if e == "domain"])
    n_data = len([e for e in edge_sources.values() if e == "data"])
    state.push_decision(
        agent="dag_expert",
        decision_type="dag_fused",
        choice=f"domain_expert_fusion ({conflict_resolution})",
        reason=(
            f"Fused {n_domain} domain-derived and {n_data} data-derived edges into "
            f"{len(validated_dag.edges)}-edge DAG; resolved {len(conflicts)} conflict(s); "
            f"applied {len(agent._forbidden_edges)} forbidden edge constraint(s)"
        ),
    )

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "n_nodes": len(validated_dag.nodes),
            "n_edges": len(validated_dag.edges),
            "n_domain_edges": n_domain,
            "n_data_edges": n_data,
            "n_conflicts_resolved": len(conflicts),
            "conflicts": conflicts,
            "forbidden_edges_applied": len(agent._forbidden_edges),
            "variable_roles_classified": len(agent._variable_roles),
            "interpretation": validated_dag.interpretation[:500],
        },
    )
