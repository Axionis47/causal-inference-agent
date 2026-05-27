"""check_outliers - IQR plus Z-score outlier detection across numeric columns."""

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "check_outliers",
    "description": "Detect outliers in specified numeric columns using IQR and Z-score methods.",
    "parameters": {
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns to check (empty for all numeric)",
        },
        "method": {
            "type": "string",
            "enum": ["iqr", "zscore", "both"],
            "description": "Detection method (default: both)",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    columns: list[str] | None = None,
    method: str = "both",
    **kwargs,
) -> ToolResult:
    df = agent._df

    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if agent._current_state.treatment_variable in columns:
            columns.remove(agent._current_state.treatment_variable)

    results = []
    for col in columns[:20]:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        data = df[col].dropna()
        if len(data) < 10:
            continue

        outlier_info: dict = {"column": col, "n": len(data)}

        if method in ["iqr", "both"]:
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            iqr_out = ((data < lower) | (data > upper)).sum()
            outlier_info["iqr_outliers"] = int(iqr_out)
            outlier_info["iqr_pct"] = float(iqr_out / len(data) * 100)

            extreme_lower = q1 - 3 * iqr
            extreme_upper = q3 + 3 * iqr
            extreme_out = ((data < extreme_lower) | (data > extreme_upper)).sum()
            outlier_info["extreme_outliers"] = int(extreme_out)

        if method in ["zscore", "both"]:
            z_scores = np.abs(stats.zscore(data))
            zscore_out = (z_scores > 3).sum()
            outlier_info["zscore_outliers"] = int(zscore_out)

        if outlier_info.get("iqr_outliers", 0) > 0 or outlier_info.get("zscore_outliers", 0) > 0:
            results.append(outlier_info)

    results.sort(key=lambda x: x.get("iqr_pct", 0), reverse=True)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "columns_analyzed": len(columns) if columns else 0,
            "outliers_detected": results[:15],
            "has_outliers": len(results) > 0,
        },
    )
