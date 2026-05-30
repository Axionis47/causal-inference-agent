"""finalize_estimation - the LLM's explicit close, with hallucination correction.

INT4: validates preferred_estimate against actual results so an LLM
that invents a number gets corrected to a real method's estimate.
"""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "finalize_estimation",
    "description": "Finalize the analysis with your conclusions. Call this when you have enough evidence.",
    "parameters": {
        "type": "object",
        "properties": {
            "preferred_method": {
                "type": "string",
                "description": "Which method's estimate you consider most credible",
            },
            "preferred_estimate": {
                "type": "number",
                "description": "The treatment effect estimate you recommend",
            },
            "confidence_level": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Your confidence in this estimate",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed reasoning for your conclusions",
            },
            "caveats": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Important caveats about the estimate",
            },
        },
        "required": ["preferred_method", "preferred_estimate", "confidence_level", "reasoning"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    preferred_method: str = "",
    preferred_estimate: float = 0.0,
    confidence_level: str = "medium",
    reasoning: str = "",
    caveats: list[str] | None = None,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="finalize_estimation",
            extra_keys=list(kwargs.keys()),
        )
    # INT4: validate preferred_estimate against actual results; correct hallucinations.
    if agent._results:
        actual_estimates = [r.estimate for r in agent._results]
        if not any(abs(preferred_estimate - e) < 1e-4 for e in actual_estimates):
            preferred_r = None
            for r in agent._results:
                if "doubly" in r.method.lower() or "aipw" in r.method.lower():
                    preferred_r = r
                    break
            if not preferred_r:
                estimates = sorted(agent._results, key=lambda x: x.estimate)
                preferred_r = estimates[len(estimates) // 2]
            original = preferred_estimate
            preferred_estimate = preferred_r.estimate
            preferred_method = preferred_r.method
            agent.logger.warning(
                "preferred_estimate_corrected",
                original=original,
                corrected_to=preferred_r.estimate,
                method=preferred_r.method,
            )

    agent._finalized = True
    agent._final_result = {
        "preferred_method": preferred_method,
        "preferred_estimate": preferred_estimate,
        "confidence_level": confidence_level,
        "reasoning": reasoning,
        "caveats": caveats or [],
        "n_methods_run": len(agent._results),
    }

    agent.logger.info(
        "estimation_finalized",
        preferred_method=preferred_method,
        preferred_estimate=preferred_estimate,
        confidence_level=confidence_level,
        n_methods=len(agent._results),
    )

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "finalized": True,
            "preferred_method": preferred_method,
            "preferred_estimate": preferred_estimate,
            "confidence_level": confidence_level,
            "reasoning": reasoning,
            "caveats": caveats or [],
        },
        metadata={"is_finish": True},
    )
