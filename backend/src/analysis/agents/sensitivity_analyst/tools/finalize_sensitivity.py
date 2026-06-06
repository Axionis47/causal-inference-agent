"""finalize_sensitivity - the LLM's explicit close; sets agent._finalized and stashes _final_result."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "finalize_sensitivity",
    "description": "Finalize the sensitivity analysis with overall robustness assessment.",
    "parameters": {
        "type": "object",
        "properties": {
            "overall_robustness": {
                "type": "string",
                "enum": ["high", "moderate", "low", "uncertain"],
                "description": "Overall assessment of causal estimate robustness",
            },
            "key_findings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key findings from sensitivity analysis",
            },
            "concerns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Any concerns about robustness",
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Recommendations for interpreting results",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed reasoning for robustness assessment",
            },
        },
        "required": ["overall_robustness", "key_findings", "reasoning"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    overall_robustness: str = "uncertain",
    key_findings: list[str] | None = None,
    reasoning: str = "",
    concerns: list[str] | None = None,
    recommendations: list[str] | None = None,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "finalize_sensitivity_ignored_kwargs",
            extra_keys=list(kwargs.keys()),
        )
    key_findings = key_findings or []
    agent._finalized = True
    agent._final_result = {
        "overall_robustness": overall_robustness,
        "key_findings": key_findings,
        "concerns": concerns or [],
        "recommendations": recommendations or [],
        "reasoning": reasoning,
        "n_analyses": len(agent._results),
    }

    agent.logger.info(
        "sensitivity_finalized",
        overall_robustness=overall_robustness,
        n_analyses=len(agent._results),
    )

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "finalized": True,
            "overall_robustness": overall_robustness,
            "key_findings": key_findings,
            "concerns": concerns or [],
            "recommendations": recommendations or [],
        },
        metadata={"is_finish": True},
    )
