"""detect_outliers - IQR and Z-score outlier detection across numeric columns."""

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "detect_outliers",
    "description": "Detect outliers using IQR and Z-score methods. Returns outlier counts and percentages.",
    "parameters": {
        "type": "object",
        "properties": {
            "variables": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Variables to check (empty for all numeric)",
            },
            "method": {
                "type": "string",
                "enum": ["iqr", "zscore", "both"],
                "description": "Outlier detection method",
            },
        },
        "required": [],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    variables: list[str] | None = None,
    method: str = "both",
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
        if agent._treatment_var in variables:
            variables.remove(agent._treatment_var)

    results = []
    for var in variables:
        if var not in df.columns or not pd.api.types.is_numeric_dtype(df[var]):
            continue
        data = df[var].dropna()
        if len(data) < 10:
            continue

        outlier_info: dict = {"variable": var, "n": len(data)}

        if method in ["iqr", "both"]:
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            iqr_outliers = ((data < lower) | (data > upper)).sum()
            outlier_info["iqr_outliers"] = int(iqr_outliers)
            outlier_info["iqr_pct"] = round(float(iqr_outliers / len(data) * 100), 1)

        if method in ["zscore", "both"]:
            z_scores = np.abs(stats.zscore(data))
            zscore_outliers = (z_scores > 3).sum()
            outlier_info["zscore_outliers"] = int(zscore_outliers)
            outlier_info["zscore_pct"] = round(float(zscore_outliers / len(data) * 100), 1)

        if outlier_info.get("iqr_outliers", 0) > 0 or outlier_info.get("zscore_outliers", 0) > 0:
            agent._outlier_results[var] = outlier_info

        results.append(outlier_info)

    with_outliers = [r for r in results if r.get("iqr_outliers", 0) > 0 or r.get("zscore_outliers", 0) > 0]

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "method": method,
            "variables_checked": len(results),
            "variables_with_outliers": len(with_outliers),
            "outlier_details": with_outliers[:10],
        },
    )
