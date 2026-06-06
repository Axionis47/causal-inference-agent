"""propose_edge - add a domain-reasoned edge to the agent's pending set."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "propose_edge",
    "description": (
        "Propose a causal edge based on domain reasoning. "
        "Specify source, target, reasoning, and confidence."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Cause variable"},
            "target": {"type": "string", "description": "Effect variable"},
            "reasoning": {"type": "string", "description": "Domain reasoning for this edge"},
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Confidence based on domain knowledge",
            },
        },
        "required": ["source", "target", "reasoning", "confidence"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    source: str = "",
    target: str = "",
    reasoning: str = "",
    confidence: str = "medium",
    **kwargs,
) -> ToolResult:
    forbidden_pairs = {(s, t) for s, t, _r in agent._forbidden_edges}
    if (source, target) in forbidden_pairs:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=f"Edge {source} -> {target} is marked as forbidden",
        )

    source_role = agent._variable_roles.get(source, "unknown")
    target_role = agent._variable_roles.get(target, "unknown")
    warnings: list[str] = []

    if target_role == "confounder" and source_role in ["treatment", "outcome"]:
        warnings.append("Warning: Treatment/outcome cannot cause a confounder (pre-treatment)")
    if source_role == "outcome" and target_role == "treatment":
        warnings.append("Warning: Outcome -> Treatment suggests reverse causality")

    edge = {
        "source": source,
        "target": target,
        "reasoning": reasoning,
        "confidence": confidence,
        "source_type": "domain",
        "warnings": warnings,
    }
    agent._domain_edges.append(edge)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "edge_added": f"{source} -> {target}",
            "confidence": confidence,
            "warnings": warnings,
            "total_domain_edges": len(agent._domain_edges),
        },
    )
