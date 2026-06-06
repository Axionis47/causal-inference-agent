"""compute_trimmed_stats - sample / overlap statistics after trimming extreme PS."""

import numpy as np

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "compute_trimmed_stats",
    "description": "Compute statistics after trimming extreme propensity scores.",
    "parameters": {
        "lower_bound": {
            "type": "number",
            "description": "Lower bound for trimming (e.g., 0.01)",
        },
        "upper_bound": {
            "type": "number",
            "description": "Upper bound for trimming (e.g., 0.99)",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    lower_bound: float = 0.01,
    upper_bound: float = 0.99,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="compute_trimmed_stats",
            extra_keys=list(kwargs.keys()),
        )
    if agent._propensity_scores is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            error="No propensity scores estimated yet.",
        )

    ps = agent._propensity_scores
    T = agent._treatment

    keep = (ps >= lower_bound) & (ps <= upper_bound)
    n_trimmed = int(np.sum(~keep))
    pct_trimmed = float((n_trimmed / len(ps)) * 100)

    ps_trimmed = ps[keep]
    T_trimmed = T[keep]

    n_treated_trimmed = int(np.sum(T_trimmed == 1))
    n_control_trimmed = int(np.sum(T_trimmed == 0))

    ps_treated = ps_trimmed[T_trimmed == 1]
    ps_control = ps_trimmed[T_trimmed == 0]

    overlap_min = max(ps_control.min(), ps_treated.min())
    overlap_max = min(ps_control.max(), ps_treated.max())
    pct_overlap = float(
        np.mean((ps_trimmed >= overlap_min) & (ps_trimmed <= overlap_max)) * 100
    )

    assessment = "Trimming improves overlap" if pct_overlap > 90 else "Consider different bounds"

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "bounds": [lower_bound, upper_bound],
            "n_trimmed": n_trimmed,
            "pct_trimmed": pct_trimmed,
            "remaining_sample": len(ps_trimmed),
            "remaining_treated": n_treated_trimmed,
            "remaining_control": n_control_trimmed,
            "ps_treated_range": [float(ps_treated.min()), float(ps_treated.max())],
            "ps_control_range": [float(ps_control.min()), float(ps_control.max())],
            "pct_overlap": pct_overlap,
            "assessment": assessment,
        },
    )
