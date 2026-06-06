"""flag_uncertainty tool: record a metadata or causal-interpretation uncertainty."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "flag_uncertainty",
    "description": (
        "Flag something you're uncertain about that downstream agents "
        "should be aware of. This helps them be cautious."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "issue": {
                "type": "string",
                "description": "What you're uncertain about",
            },
            "impact": {
                "type": "string",
                "description": "How this uncertainty might affect causal analysis",
            },
        },
        "required": ["issue", "impact"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    issue: str = "",
    impact: str = "",
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="flag_uncertainty", extra_keys=list(kwargs.keys()))

    uncertainty = {"issue": issue, "impact": impact}
    agent._uncertainties.append(uncertainty)

    agent.logger.info("uncertainty_flagged", issue=issue)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "recorded": True,
            "uncertainty": uncertainty,
            "total_uncertainties": len(agent._uncertainties),
        },
    )
