"""check_covariate_balance - SMDs per covariate, with shared treatment binarization."""

import numpy as np

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "check_covariate_balance",
    "description": "Check balance of covariates between treatment and control groups. Returns standardized mean differences.",
    "parameters": {
        "type": "object",
        "properties": {
            "covariates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of covariate names to check. If empty, checks all available covariates.",
            },
        },
        "required": [],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    covariates: list[str] | None = None,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="check_covariate_balance",
            extra_keys=list(kwargs.keys()),
        )
    df = agent._df
    T = df[agent._treatment_var].values.astype(float)

    # MED3: Binarize using shared threshold (stored in state for PS diagnostics).
    if len(np.unique(T)) > 2:
        threshold = np.median(T)
        T = (T > threshold).astype(int)
        if agent._current_state and agent._current_state.treatment_binarization_threshold is None:
            agent._current_state.treatment_binarization_threshold = float(threshold)

    if not covariates:
        covariates = agent._covariates[:15]

    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    covariates = [c for c in covariates if c in numeric_cols]

    balance_results: list[dict] = []
    imbalanced: list[str] = []

    for cov in covariates:
        if cov not in df.columns:
            continue

        try:
            x = df[cov].values.astype(float)
            treated_mean = np.nanmean(x[T == 1])
            control_mean = np.nanmean(x[T == 0])

            if np.isnan(treated_mean) or np.isnan(control_mean):
                continue

            pooled_std = np.sqrt(
                (np.nanvar(x[T == 1]) + np.nanvar(x[T == 0])) / 2
            )

            if pooled_std > 0 and not np.isnan(pooled_std):
                smd = (treated_mean - control_mean) / pooled_std
            else:
                smd = 0

            if abs(smd) < 0.1:
                status = "BALANCED"
            elif abs(smd) < 0.25:
                status = "MODERATE"
            else:
                status = "IMBALANCED"

            balance_results.append({
                "covariate": cov,
                "smd": round(smd, 3),
                "status": status,
            })

            if abs(smd) >= 0.1:
                imbalanced.append(cov)
        except Exception:
            logger.debug("covariate_balance_check_skipped", covariate=cov, exc_info=True)

    recommendation = (
        "Adjustment needed for imbalanced covariates"
        if imbalanced else "Good balance, simple methods may suffice"
    )

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "covariates_checked": len(balance_results),
            "imbalanced_count": len(imbalanced),
            "imbalanced_covariates": imbalanced,
            "balance_details": balance_results[:10],
            "recommendation": recommendation,
        },
    )
