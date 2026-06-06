"""get_estimates_summary - read state.treatment_effects so the LLM can decide what to stress-test."""

import numpy as np

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "get_estimates_summary",
    "description": "Get summary of current treatment effect estimates to understand what needs sensitivity testing.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="get_estimates_summary",
            extra_keys=list(kwargs.keys()),
        )
    if not agent._current_state.treatment_effects:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="No treatment effects estimated yet.",
        )

    estimates_data = []
    for i, effect in enumerate(agent._current_state.treatment_effects):
        estimates_data.append({
            "index": i,
            "method": effect.method,
            "estimand": effect.estimand,
            "estimate": round(effect.estimate, 4),
            "std_error": round(effect.std_error, 4),
            "ci": [round(effect.ci_lower, 4), round(effect.ci_upper, 4)],
            "p_value": round(effect.p_value, 4) if effect.p_value else None,
        })

    estimates = [e.estimate for e in agent._current_state.treatment_effects]
    summary_stats = None
    if len(estimates) > 1:
        summary_stats = {
            "mean": round(float(np.mean(estimates)), 4),
            "std": round(float(np.std(estimates)), 4),
            "range": [round(min(estimates), 4), round(max(estimates), 4)],
            "all_same_sign": all(e > 0 for e in estimates) or all(e < 0 for e in estimates),
        }

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "n_estimates": len(estimates_data),
            "estimates": estimates_data,
            "cross_method_summary": summary_stats,
        },
    )
