"""mark_forbidden_edge - block a pairing from any future fused DAG."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "mark_forbidden_edge",
    "description": (
        "Mark an edge as forbidden based on domain logic "
        "(e.g., outcome cannot cause treatment)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "target": {"type": "string"},
            "reason": {"type": "string", "description": "Why this edge is impossible"},
        },
        "required": ["source", "target", "reason"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    source: str = "",
    target: str = "",
    reason: str = "",
    **kwargs,
) -> ToolResult:
    agent._forbidden_edges.append((source, target, reason))
    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "forbidden": f"{source} -> {target}",
            "reason": reason,
            "total_forbidden": len(agent._forbidden_edges),
        },
    )
