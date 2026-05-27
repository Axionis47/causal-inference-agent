"""get_discovery_edges - read the upstream causal_discovery DAG."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "get_discovery_edges",
    "description": (
        "Get edges from data-driven causal discovery. "
        "Use to compare with domain-proposed edges."
    ),
    "parameters": {"type": "object", "properties": {}},
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    # Read discovered_dag (NOT proposed_dag) so we always see the
    # data-driven result, never our own in-progress refined_dag.
    if not state.discovered_dag:
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "available": False,
                "message": "No data-driven DAG available. Using domain knowledge only.",
            },
        )

    dag = state.discovered_dag
    discovery_edges: list[dict] = []
    for edge in dag.edges:
        edge_info = {
            "source": edge.source,
            "target": edge.target,
            "edge_type": edge.edge_type,
            "confidence": edge.confidence,
            "source_type": "data_discovery",
        }
        discovery_edges.append(edge_info)
        agent._data_edges.append(edge_info)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "discovery_method": dag.discovery_method,
            "n_edges": len(discovery_edges),
            "edges": discovery_edges[:20],
            "interpretation": dag.interpretation,
        },
    )
