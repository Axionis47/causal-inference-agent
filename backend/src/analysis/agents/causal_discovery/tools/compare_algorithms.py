"""compare_algorithms - common-edge / unique-edge summary across what's been run."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "compare_algorithms",
    "description": "Compare results from different algorithms that have been run. Identifies common and divergent edges.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    if len(agent._discovered_graphs) < 2:
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "n_algorithms": len(agent._discovered_graphs),
                "message": f"Only {len(agent._discovered_graphs)} algorithm(s) run. Run more algorithms to compare.",
            },
        )

    comparison = []
    for alg, dag in agent._discovered_graphs.items():
        directed = len([e for e in dag.edges if e.edge_type == "directed"])
        undirected = len([e for e in dag.edges if e.edge_type == "undirected"])
        has_edge = False
        if agent._treatment_var and agent._outcome_var:
            has_edge = any(
                e.source == agent._treatment_var and e.target == agent._outcome_var
                for e in dag.edges
            )
        comparison.append({
            "algorithm": alg.upper(),
            "n_nodes": len(dag.nodes),
            "n_directed": directed,
            "n_undirected": undirected,
            "treatment_outcome_edge": has_edge,
        })

    common_edges = []
    unique_edges: dict = {}
    if len(agent._discovered_graphs) >= 2:
        algs = list(agent._discovered_graphs.keys())
        dag1 = agent._discovered_graphs[algs[0]]
        dag2 = agent._discovered_graphs[algs[1]]

        edges1 = {(e.source, e.target) for e in dag1.edges if e.edge_type == "directed"}
        edges2 = {(e.source, e.target) for e in dag2.edges if e.edge_type == "directed"}

        common = edges1 & edges2
        only_first = edges1 - edges2
        only_second = edges2 - edges1

        common_edges = [{"source": s, "target": t} for s, t in list(common)[:10]]
        unique_edges = {algs[0]: len(only_first), algs[1]: len(only_second)}

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "n_algorithms": len(agent._discovered_graphs),
            "comparison": comparison,
            "common_edges": common_edges,
            "unique_edge_counts": unique_edges,
            "recommendation": "Prefer algorithms that agree on treatment-outcome relationships",
        },
    )
