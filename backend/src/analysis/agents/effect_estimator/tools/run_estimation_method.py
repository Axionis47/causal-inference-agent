"""run_estimation_method - dispatch one estimation method through the unified engine.

Gates PS-dependent methods on overlap quality (L6) and per-arm
sample size (L8). Caches result on agent._results, agent._last_method_result.
"""

import numpy as np

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

from ..estimation_methods import run_method
from ..helpers import extract_key_diagnostics

logger = get_logger(__name__)

SCHEMA = {
    "name": "run_estimation_method",
    "description": "Run a specific causal estimation method and get the treatment effect estimate.",
    "parameters": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": [
                    "ols", "ipw", "aipw", "matching",
                    "s_learner", "t_learner", "x_learner",
                    "causal_forest", "double_ml",
                    "did", "iv", "rdd",
                ],
                "description": "The estimation method to run",
            },
            "covariates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Covariates to adjust for. If empty, uses discovered confounders.",
            },
        },
        "required": ["method"],
    },
}

_PS_METHODS = {"ipw", "aipw", "psm", "matching"}


async def handle(
    agent,
    state: AnalysisState,
    method: str = "",
    covariates: list[str] | None = None,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="run_estimation_method",
            extra_keys=list(kwargs.keys()),
        )
    if not covariates:
        covariates = agent._covariates

    valid_covs = [c for c in covariates if c in agent._df.columns]

    method_lower = method.lower().replace("-", "_").replace(" ", "_")

    # L6: Block PS-dependent methods when overlap is poor.
    if agent._overlap_quality == "POOR" and method_lower in _PS_METHODS:
        state.push_decision(
            agent="effect_estimator",
            decision_type="method_rejected",
            choice=method_lower,
            reason=(
                f"PS overlap is POOR, blocking {method}. Only OLS and Double ML are valid "
                "with poor propensity score overlap."
            ),
            alternatives=["ols", "double_ml"],
        )
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=(
                f"Skipping {method} due to positivity violation (overlap quality: POOR). "
                "Only OLS and Double ML are valid with poor propensity score overlap."
            ),
        )

    # L8: Per-group sample size enforcement for PS methods.
    if method_lower in _PS_METHODS:
        T = agent._df[agent._treatment_var].values.astype(float)
        if len(np.unique(T)) > 2:
            threshold = getattr(state, "treatment_binarization_threshold", None) or np.median(T)
            T = (T > threshold).astype(int)
        n_treated = int(np.sum(T == 1))
        n_control = int(np.sum(T == 0))
        min_arm = min(n_treated, n_control)
        if min_arm < 30:
            state.push_decision(
                agent="effect_estimator",
                decision_type="method_rejected",
                choice=method_lower,
                reason=(
                    f"Smallest arm has {min_arm} samples, but {method} requires 30+ per arm "
                    f"(treated={n_treated}, control={n_control})."
                ),
                alternatives=["ols"],
            )
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=(
                    f"Insufficient samples in smaller group ({min_arm}) for {method}. "
                    f"Need at least 30 per arm. (treated={n_treated}, control={n_control})"
                ),
            )

    try:
        result = run_method(
            method, agent._treatment_var, agent._outcome_var,
            valid_covs, agent._df, agent._current_state,
        )
        if result:
            agent._results.append(result)
            agent._last_method_result = result

            diagnostics_summary = extract_key_diagnostics(result.method, result.diagnostics)

            if result.p_value:
                reason = (
                    f"Estimate={result.estimate:.4f}, SE={result.std_error:.4f}, "
                    f"95% CI=[{result.ci_lower:.4f}, {result.ci_upper:.4f}], p={result.p_value:.4f}"
                )
            else:
                reason = (
                    f"Estimate={result.estimate:.4f}, SE={result.std_error:.4f}, "
                    f"95% CI=[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
                )

            state.push_decision(
                agent="effect_estimator",
                decision_type="method_succeeded",
                choice=result.method,
                reason=reason,
            )

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "method": result.method,
                    "estimand": result.estimand,
                    "estimate": round(result.estimate, 4),
                    "std_error": round(result.std_error, 4),
                    "ci_lower": round(result.ci_lower, 4),
                    "ci_upper": round(result.ci_upper, 4),
                    "p_value": round(result.p_value, 4) if result.p_value else None,
                    "assumptions": result.assumptions_tested,
                    "diagnostics_summary": diagnostics_summary,
                    "details": result.details,
                },
            )
        else:
            state.push_decision(
                agent="effect_estimator",
                decision_type="method_failed",
                choice=method_lower,
                reason=f"Method {method} returned no result (insufficient data or inapplicable).",
            )
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Method {method} failed to produce results.",
            )
    except Exception as e:
        state.push_decision(
            agent="effect_estimator",
            decision_type="method_failed",
            choice=method_lower,
            reason=f"Method {method} raised exception: {str(e)}",
        )
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=f"Method {method} failed with error: {str(e)}",
        )
