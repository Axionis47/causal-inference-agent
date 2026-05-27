"""inspect_graph - structured view of the latest discovered DAG."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "inspect_graph",
    "description": "Inspect the most recently discovered graph structure in detail. Shows edges, treatment/outcome connections, potential confounders.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    if not agent._current_graph:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="No graph discovered yet. Run a discovery algorithm first.",
        )

    dag = agent._current_graph
    directed = [e for e in dag.edges if e.edge_type == "directed"]
    undirected = [e for e in dag.edges if e.edge_type == "undirected"]

    directed_edges = []
    for e in directed[:20]:
        edge_info: dict = {"source": e.source, "target": e.target}
        if e.confidence:
            edge_info["confidence"] = round(e.confidence, 2)
        directed_edges.append(edge_info)

    undirected_edges = [{"var1": e.source, "var2": e.target} for e in undirected[:10]]

    treatment = agent._treatment_var
    outcome = agent._outcome_var

    treatment_info = None
    if treatment and treatment in dag.nodes:
        incoming = [e.source for e in directed if e.target == treatment]
        outgoing = [e.target for e in directed if e.source == treatment]
        treatment_info = {"variable": treatment, "parents": incoming, "children": outgoing}

    outcome_info = None
    if outcome and outcome in dag.nodes:
        incoming = [e.source for e in directed if e.target == outcome]
        outgoing = [e.target for e in directed if e.source == outcome]
        outcome_info = {"variable": outcome, "parents": incoming, "children": outgoing}

    confounders = []
    if treatment and outcome:
        t_parents = {e.source for e in directed if e.target == treatment}
        o_parents = {e.source for e in directed if e.target == outcome}
        confounders = list(t_parents & o_parents)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "method": dag.discovery_method,
            "n_nodes": len(dag.nodes),
            "nodes": dag.nodes[:15] + (["..."] if len(dag.nodes) > 15 else []),
            "n_directed": len(directed),
            "n_undirected": len(undirected),
            "directed_edges": directed_edges,
            "undirected_edges": undirected_edges,
            "treatment": treatment_info,
            "outcome": outcome_info,
            "potential_confounders": confounders,
        },
    )
