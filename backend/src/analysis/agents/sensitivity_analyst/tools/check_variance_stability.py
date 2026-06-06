"""check_variance_stability - bootstrap-vs-reported SE consistency check."""

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
    "name": "check_variance_stability",
    "description": "Check if standard errors are stable via bootstrap resampling.",
    "parameters": {
        "type": "object",
        "properties": {
            "n_bootstrap": {
                "type": "integer",
                "description": "Number of bootstrap iterations (default: 200)",
            },
        },
        "required": [],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    n_bootstrap: int = 200,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="check_variance_stability",
            extra_keys=list(kwargs.keys()),
        )
    from sklearn.linear_model import LinearRegression

    df = agent._df
    T_col, Y_col = agent._resolve_treatment_outcome()

    if not T_col or not Y_col:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Treatment or outcome not identified.",
        )

    T = df[T_col].values
    Y = df[Y_col].values

    mask = ~(np.isnan(T) | np.isnan(Y))
    T = T[mask]
    Y = Y[mask]

    if len(T) < 50:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Insufficient data for bootstrap.",
        )

    np.random.seed(42)
    bootstrap_estimates = []
    n = len(T)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        T_boot = T[idx]
        Y_boot = Y[idx]

        model = LinearRegression()
        model.fit(T_boot.reshape(-1, 1), Y_boot)
        bootstrap_estimates.append(model.coef_[0])

    boot_mean = float(np.mean(bootstrap_estimates))
    boot_std = float(np.std(bootstrap_estimates))
    boot_ci = (
        float(np.percentile(bootstrap_estimates, 2.5)),
        float(np.percentile(bootstrap_estimates, 97.5)),
    )

    reported_se = (
        agent._current_state.treatment_effects[0].std_error
        if agent._current_state.treatment_effects else boot_std
    )
    se_ratio = boot_std / (reported_se + 0.001)

    if 0.8 <= se_ratio <= 1.2:
        interpretation = "STABLE: Bootstrap SE consistent with reported SE"
    elif se_ratio < 0.8:
        interpretation = "CONSERVATIVE: Reported SE may be larger than necessary"
    else:
        interpretation = "UNSTABLE: Bootstrap SE larger than reported"

    sens_result = SensitivityResult(
        method="Bootstrap Variance Check",
        robustness_value=float(min(se_ratio, 1 / se_ratio)),
        interpretation=interpretation,
        details={
            "bootstrap_se": boot_std,
            "reported_se": float(reported_se),
            "se_ratio": float(se_ratio),
        },
    )
    agent._results.append(sens_result)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "n_bootstrap": n_bootstrap,
            "boot_mean": round(boot_mean, 4),
            "boot_se": round(boot_std, 4),
            "boot_ci": [round(boot_ci[0], 4), round(boot_ci[1], 4)],
            "reported_se": round(float(reported_se), 4),
            "se_ratio": round(float(se_ratio), 2),
            "interpretation": interpretation,
        },
    )
