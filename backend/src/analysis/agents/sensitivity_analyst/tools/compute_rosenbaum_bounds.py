"""compute_rosenbaum_bounds - approximate Gamma for matched-design sensitivity."""

import numpy as np

from src.analysis.agents.base import (
    AnalysisState,
    SensitivityResult,
    ToolResult,
    ToolResultStatus,
)
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "compute_rosenbaum_bounds",
    "description": "Compute Rosenbaum bounds (Gamma) for sensitivity to unmeasured confounding in matched studies.",
    "parameters": {
        "type": "object",
        "properties": {
            "method_index": {
                "type": "integer",
                "description": "Index of the method to analyze",
            },
        },
        "required": [],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    method_index: int = 0,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="compute_rosenbaum_bounds",
            extra_keys=list(kwargs.keys()),
        )
    if method_index >= len(agent._current_state.treatment_effects):
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=f"Invalid method index {method_index}",
        )

    effect = agent._current_state.treatment_effects[method_index]
    estimate = effect.estimate
    se = effect.std_error

    z_score = abs(estimate / se) if se > 0 else 0
    gamma = 1 + z_score / 2

    if np.isnan(gamma) or np.isinf(gamma):
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Cannot compute Rosenbaum bounds: standard error is NaN or Inf.",
        )

    if gamma >= 3:
        interpretation = "VERY ROBUST: Would need very strong hidden bias"
    elif gamma >= 2:
        interpretation = "MODERATELY ROBUST: Hidden bias would need to double odds"
    elif gamma >= 1.5:
        interpretation = "SOMEWHAT SENSITIVE: Moderate hidden bias could explain effect"
    else:
        interpretation = "SENSITIVE: Small hidden bias could explain effect"

    sens_result = SensitivityResult(
        method="Rosenbaum Bounds (approximate)",
        robustness_value=float(gamma),
        interpretation=f"Gamma = {gamma:.2f}: {interpretation}",
        details={"gamma": float(gamma), "z_score": float(z_score)},
    )
    agent._results.append(sens_result)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "gamma": round(float(gamma), 2),
            "z_score": round(float(z_score), 2),
            "interpretation": interpretation,
        },
    )
