"""mark_immutable tool: add a variable to the immutable list (no duplicates)."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "mark_immutable",
    "description": (
        "Mark a variable as immutable - it cannot be caused by other variables. "
        "Examples: age, sex, race, birth year, country of origin."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "variable": {
                "type": "string",
                "description": "The immutable variable",
            },
            "reason": {
                "type": "string",
                "description": "Why this variable is immutable",
            },
        },
        "required": ["variable", "reason"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    variable: str = "",
    reason: str = "",
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="mark_immutable", extra_keys=list(kwargs.keys()))

    if variable not in agent._immutable_vars:
        agent._immutable_vars.append(variable)

    agent.logger.info("variable_marked_immutable", variable=variable)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "variable": variable,
            "reason": reason,
            "total_immutable": len(agent._immutable_vars),
        },
    )
