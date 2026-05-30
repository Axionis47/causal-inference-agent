"""compute_e_value - VanderWeele E-value for sensitivity to unmeasured confounding."""

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
    "name": "compute_e_value",
    "description": "Compute E-value for sensitivity to unmeasured confounding. This quantifies how strong an unmeasured confounder would need to be to explain away the observed effect.",
    "parameters": {
        "type": "object",
        "properties": {
            "method_index": {
                "type": "integer",
                "description": "Index of the method to analyze (0 for first/primary estimate)",
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
            tool="compute_e_value",
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

    if se <= 0 or np.isnan(se):
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=(
                "Cannot compute E-value: standard error is zero or NaN. "
                "The estimate may be unreliable."
            ),
        )

    # CR3: Proper E-value conversion using VanderWeele (2017) approximation.
    # For continuous outcomes: convert to Cohen's d, then RR ≈ exp(0.91 * d).
    n_total = len(agent._df) if agent._df is not None else 1000
    pooled_sd = se * np.sqrt(n_total) if se > 0 else 1.0
    d = abs(estimate) / pooled_sd if pooled_sd > 0 else abs(estimate)
    rr = max(1.01, float(np.exp(0.91 * d)))

    e_value = rr + np.sqrt(rr * (rr - 1))

    ci_bound = effect.ci_lower if estimate > 0 else effect.ci_upper
    if (estimate > 0 and ci_bound > 0) or (estimate < 0 and ci_bound < 0):
        d_ci = abs(ci_bound) / pooled_sd if pooled_sd > 0 else abs(ci_bound)
        rr_ci = max(1.01, float(np.exp(0.91 * d_ci)))
        e_value_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1))
    else:
        e_value_ci = 1.0

    if e_value >= 3 and e_value_ci >= 1.5:
        interpretation = "ROBUST: Would need very strong unmeasured confounding"
        robustness_level = "high"
    elif e_value >= 2 and e_value_ci >= 1.2:
        interpretation = "MODERATELY ROBUST: Moderate confounding strength needed"
        robustness_level = "moderate"
    elif e_value >= 1.5:
        interpretation = "SOMEWHAT SENSITIVE: Relatively weak confounding could explain"
        robustness_level = "low"
    else:
        interpretation = "SENSITIVE: Even weak confounding could explain the effect"
        robustness_level = "very low"

    sens_result = SensitivityResult(
        method="E-value",
        robustness_value=float(e_value),
        interpretation=f"E-value = {e_value:.2f} (CI: {e_value_ci:.2f}): {interpretation}",
        details={
            "e_value_point": float(e_value),
            "e_value_ci": float(e_value_ci),
            "approximate_rr": float(rr),
            "robustness_level": robustness_level,
        },
    )
    agent._results.append(sens_result)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "method_analyzed": effect.method,
            "e_value_point": round(float(e_value), 2),
            "e_value_ci": round(float(e_value_ci), 2),
            "approximate_rr": round(float(rr), 2),
            "robustness_level": robustness_level,
            "interpretation": interpretation,
        },
    )
