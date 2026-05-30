"""estimate_propensity_scores - fit a logistic PS model on the given covariates."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "estimate_propensity_scores",
    "description": "Estimate propensity scores using logistic regression with specified covariates.",
    "parameters": {
        "covariates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Covariates to include in PS model. If empty, uses all available confounders.",
        },
        "add_interactions": {
            "type": "boolean",
            "description": "Whether to add pairwise interactions",
        },
        "add_polynomials": {
            "type": "boolean",
            "description": "Whether to add quadratic terms for numeric covariates",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    covariates: list[str] | None = None,
    add_interactions: bool = False,
    add_polynomials: bool = False,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="estimate_propensity_scores",
            extra_keys=list(kwargs.keys()),
        )
    df = agent._df
    cov_list = covariates if covariates else agent._confounders

    cov_list = [c for c in cov_list if c in df.columns]
    if not cov_list:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            error="No valid covariates specified.",
        )

    X = df[cov_list].values
    T = df[agent._treatment_var].values

    mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(T))
    X = X[mask]
    T = T[mask]

    if len(X) < 50:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            error="Insufficient data after removing missing values.",
        )

    if add_polynomials:
        X_poly = []
        for i in range(X.shape[1]):
            X_poly.append(X[:, i])
            X_poly.append(X[:, i] ** 2)
        X = np.column_stack(X_poly)

    if add_interactions and X.shape[1] <= 10:
        interactions = []
        for i in range(min(X.shape[1], 5)):
            for j in range(i + 1, min(X.shape[1], 5)):
                interactions.append(X[:, i] * X[:, j])
        if interactions:
            X = np.column_stack([X] + interactions)

    try:
        model = LogisticRegression(max_iter=1000, random_state=42, penalty="l2", C=1.0)
        model.fit(X, T)
        ps = model.predict_proba(X)[:, 1]
    except Exception as e:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            error=f"Model fitting failed: {str(e)}",
        )

    agent._propensity_scores = ps
    agent._treatment = T
    agent._mask = mask

    n_treated = int(np.sum(T == 1))
    n_control = int(np.sum(T == 0))

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "covariates_used": len(cov_list),
            "add_interactions": add_interactions,
            "add_polynomials": add_polynomials,
            "n_treated": n_treated,
            "n_control": n_control,
            "ps_treated_range": [float(ps[T == 1].min()), float(ps[T == 1].max())],
            "ps_treated_mean": float(ps[T == 1].mean()),
            "ps_control_range": [float(ps[T == 0].min()), float(ps[T == 0].max())],
            "ps_control_mean": float(ps[T == 0].mean()),
            "ps_near_zero": int(np.sum(ps < 0.01)),
            "ps_near_one": int(np.sum(ps > 0.99)),
        },
    )
