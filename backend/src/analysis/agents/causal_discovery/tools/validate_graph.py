"""validate_graph - sanity checks: T-O path, reverse causation, density, isolation."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from ..helpers import check_path

SCHEMA = {
    "name": "validate_graph",
    "description": "Validate the discovered graph for causal sensibility. Checks treatment-outcome path, reverse causation, density, and isolated nodes.",
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
            error="No graph to validate. Run a discovery algorithm first.",
        )

    dag = agent._current_graph
    issues: list[str] = []
    warnings: list[str] = []

    treatment = agent._treatment_var
    outcome = agent._outcome_var

    if treatment and treatment not in dag.nodes:
        issues.append(f"Treatment variable '{treatment}' not in graph nodes")
    if outcome and outcome not in dag.nodes:
        issues.append(f"Outcome variable '{outcome}' not in graph nodes")

    has_direct_edge = False
    has_path = False
    if treatment and outcome and treatment in dag.nodes and outcome in dag.nodes:
        has_direct_edge = any(e.source == treatment and e.target == outcome for e in dag.edges)
        has_path = check_path(dag, treatment, outcome)
        if not has_direct_edge and not has_path:
            warnings.append("No path from treatment to outcome - may indicate no causal effect")

    reverse_edge = False
    if treatment and outcome:
        reverse_edge = any(e.source == outcome and e.target == treatment for e in dag.edges)
        if reverse_edge:
            issues.append(f"Reverse edge {outcome} -> {treatment} detected - may be spurious")

    n_nodes = len(dag.nodes)
    n_edges = len(dag.edges)
    max_edges = n_nodes * (n_nodes - 1) / 2
    density = n_edges / max_edges if max_edges > 0 else 0

    if density > 0.5:
        warnings.append("Graph is very dense - may include spurious edges")
    elif density < 0.05 and n_nodes > 5:
        warnings.append("Graph is very sparse - may be missing edges")

    connected: set = set()
    for e in dag.edges:
        connected.add(e.source)
        connected.add(e.target)
    isolated = list(set(dag.nodes) - connected)
    if isolated:
        warnings.append(f"Isolated nodes (no edges): {isolated[:5]}")

    if not issues and not warnings:
        validation_status = "passed"
    elif not issues:
        validation_status = "passed_with_warnings"
    else:
        validation_status = "issues_found"

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "validation_status": validation_status,
            "has_treatment_outcome_edge": has_direct_edge,
            "has_treatment_outcome_path": has_path,
            "reverse_causation": reverse_edge,
            "density": round(density * 100, 1),
            "n_isolated": len(isolated),
            "issues": issues,
            "warnings": warnings,
        },
    )
