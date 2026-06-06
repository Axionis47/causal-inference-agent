"""compute_correlations - correlation matrix and high-correlation pairs."""

import numpy as np
import pandas as pd

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "compute_correlations",
    "description": "Compute correlation matrix and identify high correlations. Use to check for multicollinearity.",
    "parameters": {
        "type": "object",
        "properties": {
            "variables": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Variables to include (empty for all numeric)",
            },
            "method": {
                "type": "string",
                "enum": ["pearson", "spearman"],
                "description": "Correlation method",
            },
            "threshold": {
                "type": "number",
                "description": "Threshold for flagging high correlations (default: 0.7)",
            },
        },
        "required": [],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    variables: list[str] | None = None,
    method: str = "pearson",
    threshold: float = 0.7,
    **kwargs,
) -> ToolResult:
    if agent._df is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Dataset not loaded",
        )

    df = agent._df

    if not variables:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()
    variables = [v for v in variables if v in df.columns and pd.api.types.is_numeric_dtype(df[v])]

    if len(variables) < 2:
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"error": "Need at least 2 numeric variables"},
        )

    corr_matrix = df[variables].corr(method=method)

    high_corrs = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    high_corrs.append({
                        "var1": col1,
                        "var2": col2,
                        "correlation": round(float(corr_val), 3),
                    })

    high_corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    agent._correlation_results = {
        "method": method,
        "n_variables": len(variables),
        "high_correlations": high_corrs,
    }
    agent._eda_result.correlation_matrix = corr_matrix.to_dict()
    agent._eda_result.high_correlations = high_corrs

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "method": method,
            "threshold": threshold,
            "n_variables": len(variables),
            "high_correlations_count": len(high_corrs),
            "high_correlations": high_corrs[:10],
        },
    )
