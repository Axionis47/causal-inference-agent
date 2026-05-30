"""get_covariate_summary - per-arm summary stats and SMD for one covariate."""

import numpy as np

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "get_covariate_summary",
    "description": "Get summary statistics for a specific covariate to understand imbalance.",
    "parameters": {
        "covariate": {
            "type": "string",
            "description": "Name of the covariate to examine",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    covariate: str = "",
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="get_covariate_summary",
            extra_keys=list(kwargs.keys()),
        )
    if not covariate or covariate not in agent._df.columns:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            error=f"Covariate '{covariate}' not found.",
        )

    df = agent._df
    T = df[agent._treatment_var].values
    x = df[covariate].values

    valid = ~(np.isnan(x) | np.isnan(T))
    x = x[valid]
    T = T[valid]

    treated_vals = x[T == 1]
    control_vals = x[T == 0]

    pooled_var = (np.var(treated_vals) + np.var(control_vals)) / 2
    smd = (
        (np.mean(treated_vals) - np.mean(control_vals)) / np.sqrt(pooled_var)
        if pooled_var > 0 else 0
    )

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "covariate": covariate,
            "treated": {
                "n": len(treated_vals),
                "mean": float(np.mean(treated_vals)),
                "std": float(np.std(treated_vals)),
                "median": float(np.median(treated_vals)),
                "range": [float(np.min(treated_vals)), float(np.max(treated_vals))],
            },
            "control": {
                "n": len(control_vals),
                "mean": float(np.mean(control_vals)),
                "std": float(np.std(control_vals)),
                "median": float(np.median(control_vals)),
                "range": [float(np.min(control_vals)), float(np.max(control_vals))],
            },
            "mean_difference": float(np.mean(treated_vals) - np.mean(control_vals)),
            "smd": float(smd),
        },
    )
