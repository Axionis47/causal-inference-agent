"""run_specification_curve - estimate stability across covariate-subset specifications."""

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
    "name": "run_specification_curve",
    "description": "Run specification curve analysis to see how estimates vary across different model specifications.",
    "parameters": {
        "type": "object",
        "properties": {
            "n_specifications": {
                "type": "integer",
                "description": "Number of specifications to try (default: 10)",
            },
        },
        "required": [],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    n_specifications: int = 10,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="run_specification_curve",
            extra_keys=list(kwargs.keys()),
        )
    from sklearn.linear_model import LinearRegression

    df = agent._df
    T_col, Y_col = agent._resolve_treatment_outcome()

    if not T_col or not Y_col:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Treatment or outcome variable not identified.",
        )

    if T_col not in df.columns or Y_col not in df.columns:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Treatment or outcome variable not in dataset.",
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
            error="Insufficient data for specification curve.",
        )

    covariates = []
    if agent._current_state.data_profile:
        covariates = [
            c for c in agent._current_state.data_profile.potential_confounders
            if c in df.columns and c != T_col and c != Y_col
        ][:15]

    estimates = []
    specs = []

    model = LinearRegression()
    model.fit(T.reshape(-1, 1), Y)
    estimates.append(model.coef_[0])
    specs.append("No controls")

    skipped = 0
    for i in range(min(n_specifications - 1, len(covariates))):
        cov_subset = covariates[: i + 1]
        try:
            cov_data = df[cov_subset].values[mask]
            valid = ~np.any(np.isnan(cov_data), axis=1)
            X_cov = np.column_stack([T[valid], cov_data[valid]])
            Y_cov = Y[valid]

            if len(X_cov) > 30:
                model = LinearRegression()
                model.fit(X_cov, Y_cov)
                estimates.append(model.coef_[0])
                specs.append(f"+{cov_subset[-1]}")
        except Exception as e:
            agent.logger.debug("spec_curve_iteration_failed", iteration=i, error=str(e))
            skipped += 1

    if len(estimates) < 2:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Could not compute multiple specifications.",
        )

    est_mean = float(np.mean(estimates))
    est_std = float(np.std(estimates))
    all_same_sign = all(e > 0 for e in estimates) or all(e < 0 for e in estimates)
    cv = est_std / abs(est_mean) if est_mean != 0 else float("inf")

    if cv < 0.2 and all_same_sign:
        interpretation = "HIGHLY STABLE: Estimates consistent across specifications"
    elif cv < 0.4 and all_same_sign:
        interpretation = "MODERATELY STABLE: Some variation but same direction"
    elif all_same_sign:
        interpretation = "VARIABLE: Magnitude varies but direction consistent"
    else:
        interpretation = "UNSTABLE: Sign changes across specifications"

    sens_result = SensitivityResult(
        method="Specification Curve",
        robustness_value=float(1 - min(cv, 1)),
        interpretation=f"{interpretation} (CV={cv:.2f})",
        details={
            "n_specifications": len(estimates),
            "estimate_mean": est_mean,
            "estimate_std": est_std,
            "all_same_sign": all_same_sign,
            "cv": float(cv),
        },
    )
    agent._results.append(sens_result)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "n_specifications": len(estimates),
            "mean_estimate": round(est_mean, 4),
            "std_estimate": round(est_std, 4),
            "range": [round(min(estimates), 4), round(max(estimates), 4)],
            "all_same_sign": all_same_sign,
            "cv": round(cv, 4),
            "specifications_skipped": skipped,
            "interpretation": interpretation,
        },
    )
