"""get_data_characteristics - sample size, distributions, correlations for algorithm choice."""

import pandas as pd
from scipy import stats as sp_stats

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from ..helpers import select_columns

SCHEMA = {
    "name": "get_data_characteristics",
    "description": "Get data characteristics relevant for algorithm selection: sample size, variable count, distributions, correlations, Gaussianity. Call this early.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    if agent._df is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Dataset not loaded",
        )

    df_work, relevant_cols = select_columns(state, agent._df, agent._treatment_var, agent._outcome_var)
    if not relevant_cols:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="No relevant columns found for data characteristics analysis.",
        )

    df_subset = df_work[relevant_cols].dropna()

    distributions = []
    non_gaussian_count = 0
    for col in relevant_cols[:10]:
        data = df_subset[col].dropna()
        if len(data) > 10:
            skew = sp_stats.skew(data)
            if len(data) <= 5000:
                _, p_val = sp_stats.shapiro(data.sample(min(len(data), 5000), random_state=42))
            else:
                _, p_val = sp_stats.normaltest(data)
            is_gaussian = p_val > 0.05
            if not is_gaussian:
                non_gaussian_count += 1
            distributions.append({
                "column": col,
                "skewness": round(skew, 2),
                "gaussian": is_gaussian,
            })

    high_corrs = []
    if df_subset.empty or len(df_subset.columns) < 2:
        corr_matrix = pd.DataFrame()
    else:
        corr_matrix = df_subset.corr().abs()
    for i, c1 in enumerate(corr_matrix.columns):
        for j, c2 in enumerate(corr_matrix.columns):
            if i < j and corr_matrix.iloc[i, j] > 0.7:
                high_corrs.append({
                    "var1": c1,
                    "var2": c2,
                    "correlation": round(float(corr_matrix.iloc[i, j]), 2),
                })

    n_samples = len(df_subset)
    n_vars = len(relevant_cols)
    recommendations: list[str] = []
    if n_samples < 300:
        recommendations.append("WARNING: Small sample size - results may be unreliable")
        recommendations.append("Consider PC with higher alpha (0.1) or use simple DAG")
    elif n_vars > 15:
        recommendations.append("Many variables - consider PC (efficient for sparse graphs)")
    else:
        recommendations.append("Sample size adequate for discovery")

    if non_gaussian_count > len(relevant_cols) * 0.5:
        recommendations.append("Most variables are non-Gaussian - LiNGAM may work well")
    else:
        recommendations.append("Many Gaussian variables - avoid LiNGAM, use PC or GES")

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "n_samples": len(agent._df),
            "n_complete_cases": len(df_subset),
            "n_variables": len(relevant_cols),
            "variables": relevant_cols,
            "treatment": agent._treatment_var,
            "outcome": agent._outcome_var,
            "distributions": distributions[:10],
            "non_gaussian_pct": round(non_gaussian_count / max(len(distributions), 1) * 100, 1),
            "high_correlations": high_corrs[:5],
            "recommendations": recommendations,
        },
    )
