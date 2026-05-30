"""compare_estimates - cross-method synthesis with reliability-weighted average + MAD outliers."""

import numpy as np

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "compare_estimates",
    "description": "Compare all estimates obtained so far across methods.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

_RELIABILITY_WEIGHTS = {"high": 3, "medium": 2, "low": 1, None: 1}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="compare_estimates",
            extra_keys=list(kwargs.keys()),
        )
    if not agent._results:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="No estimates to compare yet. Run some methods first.",
        )

    estimates = [r.estimate for r in agent._results]

    mean_est = float(np.mean(estimates))
    std_est = float(np.std(estimates))
    cv = std_est / (abs(mean_est) + 1e-10)

    median_est = float(np.median(estimates))
    mad = float(np.median(np.abs(np.array(estimates) - median_est)))

    weighted_sum = 0.0
    weight_total = 0.0

    individual_estimates: list[dict] = []
    for r in agent._results:
        reliability = r.details.get("reliability", "unknown") if r.details else "unknown"
        weight = _RELIABILITY_WEIGHTS.get(reliability, 1)
        weighted_sum += r.estimate * weight
        weight_total += weight
        individual_estimates.append({
            "method": r.method,
            "estimate": round(r.estimate, 4),
            "std_error": round(r.std_error, 4),
            "reliability": reliability,
        })

    weighted_avg = weighted_sum / weight_total if weight_total > 0 else mean_est

    outliers: list[str] = []
    for r in agent._results:
        if mad > 0 and abs(r.estimate - median_est) > 3 * mad:
            outliers.append(r.method)

    if cv < 0.2:
        consistency = "CONSISTENT"
    elif cv < 0.5:
        consistency = "MODERATE"
    else:
        consistency = "INCONSISTENT"

    chosen = mean_est if consistency == "CONSISTENT" else median_est
    label = "mean" if consistency == "CONSISTENT" else "median"

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "n_methods": len(agent._results),
            "individual_estimates": individual_estimates,
            "statistics": {
                "mean": round(mean_est, 4),
                "median": round(median_est, 4),
                "weighted_mean": round(weighted_avg, 4),
                "std": round(std_est, 4),
                "mad": round(mad, 4),
                "cv": round(cv, 4),
                "range": [round(min(estimates), 4), round(max(estimates), 4)],
            },
            "consistency": consistency,
            "outlier_methods": outliers,
            "recommendation": f"Use {label} estimate: {round(chosen, 4)}",
        },
    )
