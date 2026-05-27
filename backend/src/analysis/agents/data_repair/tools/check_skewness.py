"""check_skewness - flag heavily-skewed numeric columns with a transformation hint."""

import pandas as pd
from scipy import stats

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "check_skewness",
    "description": "Check distribution skewness for numeric columns to assess need for transformation.",
    "parameters": {
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns to check (empty for all numeric)",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    columns: list[str] | None = None,
    **kwargs,
) -> ToolResult:
    import numpy as np
    df = agent._df

    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    for col in columns[:20]:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        data = df[col].dropna()
        if len(data) < 10:
            continue

        skew = float(stats.skew(data))
        kurt = float(stats.kurtosis(data))

        if abs(skew) > 1:
            results.append({
                "column": col,
                "skewness": skew,
                "kurtosis": kurt,
                "recommendation": "log_transform" if skew > 1 and data.min() > 0 else "winsorize",
            })

    results.sort(key=lambda x: abs(x["skewness"]), reverse=True)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "columns_analyzed": len(columns) if columns else 0,
            "skewed_columns": results[:15],
            "has_skewed_variables": len(results) > 0,
        },
    )
