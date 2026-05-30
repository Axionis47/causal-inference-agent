"""estimate_propensity_scores - fit PS via logistic regression + classify overlap quality.

Caches agent._propensity_scores and agent._overlap_quality so
downstream tools (run_estimation_method) can gate PS-dependent
methods on overlap.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "estimate_propensity_scores",
    "description": "Estimate propensity scores and check overlap between treatment groups.",
    "parameters": {
        "type": "object",
        "properties": {
            "covariates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Covariates to use for propensity model. If empty, uses discovered confounders or all available.",
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
            tool="estimate_propensity_scores",
            extra_keys=list(kwargs.keys()),
        )
    df = agent._df
    T = df[agent._treatment_var].values.astype(float)

    # MED3: Binarize using shared threshold.
    if len(np.unique(T)) > 2:
        threshold = np.median(T)
        T = (T > threshold).astype(int)
        if agent._current_state and agent._current_state.treatment_binarization_threshold is None:
            agent._current_state.treatment_binarization_threshold = float(threshold)

    if not covariates:
        covariates = agent._covariates[:15]

    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    valid_covs = [c for c in covariates if c in df.columns and c in numeric_cols]
    if not valid_covs:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="No valid numeric covariates for propensity model.",
        )

    X = df[valid_covs].values.astype(float)

    mask = ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]
    T_clean = T[mask]

    if len(X_clean) < 50:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Insufficient data after removing missing values.",
        )

    try:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_clean, T_clean)
        ps = model.predict_proba(X_clean)[:, 1]
    except Exception as e:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=f"Propensity model failed: {str(e)}",
        )

    agent._propensity_scores = np.full(len(T), np.nan)
    agent._propensity_scores[mask] = ps

    ps_treated = ps[T_clean == 1]
    ps_control = ps[T_clean == 0]

    overlap_min = max(ps_control.min(), ps_treated.min())
    overlap_max = min(ps_control.max(), ps_treated.max())

    in_support = float(np.mean((ps >= overlap_min) & (ps <= overlap_max)))

    if in_support > 0.9:
        overlap_quality = "GOOD"
    elif in_support > 0.7:
        overlap_quality = "MODERATE"
    else:
        overlap_quality = "POOR"

    # L6: Cache for run_estimation_method's PS-method gating.
    agent._overlap_quality = overlap_quality

    if in_support > 0.9:
        recommendation = "Good overlap supports IPW and matching methods"
    elif in_support > 0.7:
        recommendation = "Consider trimming extreme PS; AIPW may be more robust than IPW"
    else:
        recommendation = "Poor overlap is concerning - results may be extrapolation"

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "treated_ps": {
                "mean": round(float(ps_treated.mean()), 3),
                "std": round(float(ps_treated.std()), 3),
                "min": round(float(ps_treated.min()), 3),
                "max": round(float(ps_treated.max()), 3),
            },
            "control_ps": {
                "mean": round(float(ps_control.mean()), 3),
                "std": round(float(ps_control.std()), 3),
                "min": round(float(ps_control.min()), 3),
                "max": round(float(ps_control.max()), 3),
            },
            "common_support": [round(overlap_min, 3), round(overlap_max, 3)],
            "proportion_in_support": round(in_support, 3),
            "overlap_quality": overlap_quality,
            "recommendation": recommendation,
        },
    )
