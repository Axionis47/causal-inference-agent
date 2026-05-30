"""check_covariate_balance - SMD per covariate, weighted or unweighted.

Caches `agent._metrics.max_smd_weighted` on the weighted pass so the
brief's SMD_HIGH flag derivation works. The unweighted pass is
informative but not part of the contract.
"""

import numpy as np

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "check_covariate_balance",
    "description": "Check standardized mean differences for all covariates between treatment groups.",
    "parameters": {
        "weighted": {
            "type": "boolean",
            "description": "If true, compute balance after IPW weighting",
        },
        "specific_covariates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific covariates to check. If empty, checks all.",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    weighted: bool = False,
    specific_covariates: list[str] | None = None,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="check_covariate_balance",
            extra_keys=list(kwargs.keys()),
        )
    if agent._propensity_scores is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            error="No propensity scores estimated yet.",
        )

    df = agent._df
    ps = agent._propensity_scores
    T = agent._treatment
    mask = agent._mask

    cov_list = specific_covariates if specific_covariates else agent._confounders[:15]
    cov_list = [c for c in cov_list if c in df.columns]

    try:
        smd_results: dict = {}
        smd_values: list[float] = []
        imbalanced: list[str] = []
        severely_imbalanced: list[str] = []

        df_subset = df[mask].copy()

        for cov in cov_list:
            x = df_subset[cov].values

            if weighted:
                weights_t = T / ps
                weights_c = (1 - T) / (1 - ps)

                weights_t = np.clip(weights_t, 0, 10)
                weights_c = np.clip(weights_c, 0, 10)

                treated_mean = np.average(x[T == 1], weights=weights_t[T == 1])
                control_mean = np.average(x[T == 0], weights=weights_c[T == 0])
            else:
                treated_mean = np.mean(x[T == 1])
                control_mean = np.mean(x[T == 0])

            pooled_std = np.sqrt((np.var(x[T == 1]) + np.var(x[T == 0])) / 2)
            smd = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0

            smd_values.append(abs(smd))

            if abs(smd) >= 0.25:
                status = "SEVERE"
                severely_imbalanced.append(cov)
                imbalanced.append(cov)
            elif abs(smd) >= 0.1:
                status = "IMBALANCED"
                imbalanced.append(cov)
            else:
                status = "OK"

            smd_results[cov] = {"smd": float(smd), "status": status}

        mean_smd = float(np.mean(smd_values))
        max_smd = float(np.max(smd_values))
        if weighted:
            agent._metrics.max_smd_weighted = max_smd

        if mean_smd < 0.05 and len(imbalanced) == 0:
            quality = "EXCELLENT"
        elif mean_smd < 0.1 and len(severely_imbalanced) == 0:
            quality = "GOOD"
        elif mean_smd < 0.15:
            quality = "MODERATE"
        else:
            quality = "POOR"

    except Exception as e:
        agent.logger.warning("balance_computation_failed", error=str(e), exc_info=True)
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=f"Balance computation failed: {str(e)}",
        )

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "weighted": weighted,
            "covariate_balance": smd_results,
            "mean_smd": mean_smd,
            "max_smd": max_smd,
            "n_imbalanced": len(imbalanced),
            "n_severely_imbalanced": len(severely_imbalanced),
            "imbalanced_covariates": imbalanced,
            "quality": quality,
        },
    )
