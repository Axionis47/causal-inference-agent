"""finalize_diagnostics - the LLM's explicit close: writes state.ps_diagnostics + sets agent._finalized."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "finalize_diagnostics",
    "description": "Finalize PS diagnostics with recommendations. Call this when you have enough evidence.",
    "parameters": {
        "model_quality": {
            "type": "string",
            "enum": ["excellent", "good", "acceptable", "needs_improvement", "unreliable"],
            "description": "Overall quality of the PS model",
        },
        "proceed_with_analysis": {
            "type": "boolean",
            "description": "Whether to proceed with PS-based estimation",
        },
        "recommended_method": {
            "type": "string",
            "enum": ["ipw", "matching", "aipw", "doubly_robust", "other"],
            "description": "Recommended estimation method given diagnostics",
        },
        "trimming_bounds": {
            "type": "array",
            "items": {"type": "number"},
            "description": "[lower, upper] bounds for PS trimming, or null if no trimming",
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Important warnings about using PS in this analysis",
        },
        "reasoning": {
            "type": "string",
            "description": "Detailed reasoning for recommendations",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    model_quality: str = "acceptable",
    proceed_with_analysis: bool = True,
    recommended_method: str = "aipw",
    reasoning: str = "",
    trimming_bounds: list[float] | None = None,
    warnings: list[str] | None = None,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="finalize_diagnostics",
            extra_keys=list(kwargs.keys()),
        )
    agent.logger.info(
        "ps_finalizing",
        model_quality=model_quality,
        proceed_with_analysis=proceed_with_analysis,
        recommended_method=recommended_method,
    )

    result = {
        "model_quality": model_quality,
        "proceed_with_analysis": proceed_with_analysis,
        "recommended_method": recommended_method,
        "trimming_bounds": trimming_bounds,
        "warnings": warnings or [],
        "reasoning": reasoning,
    }

    state.ps_diagnostics = result
    agent._finalized = True

    return ToolResult(status=ToolResultStatus.SUCCESS, output=result)
