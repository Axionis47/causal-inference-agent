"""set_temporal_ordering tool: record an ordering narrative and seed immutables."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "set_temporal_ordering",
    "description": (
        "Record your understanding of temporal ordering - which variables "
        "came before others. This is crucial for causal inference."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ordering": {
                "type": "string",
                "description": (
                    "Description of temporal ordering "
                    "(e.g., 'Demographics at baseline, treatment assigned, then outcome measured')"
                ),
            },
            "pre_treatment_vars": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Variables measured before treatment",
            },
            "post_treatment_vars": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Variables measured after treatment",
            },
        },
        "required": ["ordering"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    ordering: str = "",
    pre_treatment_vars: list[str] | None = None,
    post_treatment_vars: list[str] | None = None,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="set_temporal_ordering", extra_keys=list(kwargs.keys()))

    agent._temporal_understanding = ordering

    if pre_treatment_vars:
        for var in pre_treatment_vars:
            if var not in agent._immutable_vars:
                agent._immutable_vars.append(var)

    agent.logger.info("temporal_ordering_set", ordering=ordering)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "ordering": ordering,
            "pre_treatment_vars": pre_treatment_vars or [],
            "post_treatment_vars": post_treatment_vars or [],
        },
    )
