"""check_collinearity - high pairwise correlations and VIF scores."""

import numpy as np

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "check_collinearity",
    "description": "Check correlations and VIF scores for covariates to detect multicollinearity.",
    "parameters": {
        "covariates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Covariates to check (empty for all numeric except treatment/outcome)",
        },
        "correlation_threshold": {
            "type": "number",
            "description": "Threshold for flagging high correlations (default: 0.8)",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    covariates: list[str] | None = None,
    correlation_threshold: float = 0.8,
    **kwargs,
) -> ToolResult:
    df = agent._df

    if not covariates:
        covariates = df.select_dtypes(include=[np.number]).columns.tolist()
        for var in [agent._current_state.treatment_variable, agent._current_state.outcome_variable]:
            if var in covariates:
                covariates.remove(var)

    covariates = [c for c in covariates if c in df.columns]

    if len(covariates) < 2:
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"message": "Not enough covariates for collinearity analysis."},
        )

    corr_matrix = df[covariates].corr().abs()
    high_corrs = []

    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > correlation_threshold:
                    high_corrs.append({
                        "var1": col1,
                        "var2": col2,
                        "correlation": float(corr_val),
                    })

    high_corrs.sort(key=lambda x: x["correlation"], reverse=True)

    vif_results = []
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X = df[covariates].dropna()
        if len(X) >= 10:
            X_with_const = np.column_stack([np.ones(len(X)), X.values])
            for i, col in enumerate(covariates):
                try:
                    vif = variance_inflation_factor(X_with_const, i + 1)
                    if vif > 5:
                        vif_results.append({"variable": col, "vif": float(vif)})
                except Exception:
                    agent.logger.debug("vif_single_var_skipped", variable=col, exc_info=True)
    except Exception:
        agent.logger.debug("vif_computation_skipped", exc_info=True)

    vif_results.sort(key=lambda x: x["vif"], reverse=True)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "covariates_analyzed": len(covariates),
            "high_correlations": high_corrs[:10],
            "high_vif_variables": vif_results[:10],
            "has_collinearity_issues": len(high_corrs) > 0 or len(vif_results) > 0,
        },
    )
