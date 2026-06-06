"""repair_outliers - winsorize, clip, log-transform, robust-scale, or remove."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from ..helpers import count_outliers

SCHEMA = {
    "name": "repair_outliers",
    "description": "Handle outliers in specified columns using chosen strategy.",
    "parameters": {
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns to repair",
        },
        "strategy": {
            "type": "string",
            "enum": ["winsorize", "clip", "log_transform", "robust_scale", "remove"],
            "description": "Outlier handling strategy",
        },
        "percentiles": {
            "type": "array",
            "items": {"type": "number"},
            "description": "For winsorize: [lower, upper] percentiles (default: [1, 99])",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    columns: list[str] | None = None,
    strategy: str = "clip",
    percentiles: list[float] | None = None,
    **kwargs,
) -> ToolResult:
    columns = columns or []
    if percentiles is None:
        percentiles = [1, 99]

    df = agent._df

    if agent._current_state.treatment_variable in columns:
        columns = [c for c in columns if c != agent._current_state.treatment_variable]
        if not columns:
            return ToolResult(status=ToolResultStatus.ERROR, error="Cannot modify treatment variable.")

    columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not columns:
        return ToolResult(status=ToolResultStatus.ERROR, error="No valid numeric columns to repair.")

    changes = []
    try:
        for col in columns:
            data = df[col].dropna()
            if len(data) < 10:
                continue

            before_outliers = count_outliers(data)

            if strategy == "winsorize":
                lower_p, upper_p = percentiles[0], percentiles[1]
                lower = df[col].quantile(lower_p / 100)
                upper = df[col].quantile(upper_p / 100)
                df[col] = df[col].clip(lower=lower, upper=upper)
            elif strategy == "clip":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                df[col] = df[col].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
            elif strategy == "log_transform":
                min_val = df[col].min()
                df[col] = np.log1p(df[col] - min_val + 1) if min_val <= 0 else np.log(df[col])
            elif strategy == "robust_scale":
                scaler = RobustScaler()
                df[col] = scaler.fit_transform(df[[col]])
            elif strategy == "remove":
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                mask = (df[col] >= q1 - 3 * iqr) & (df[col] <= q3 + 3 * iqr)
                rows_before = len(df)
                df = df[mask | df[col].isnull()]
                changes.append({"column": col, "removed": rows_before - len(df)})
                continue

            after_outliers = count_outliers(df[col].dropna())
            changes.append({"column": col, "before": before_outliers, "after": after_outliers})

        agent._df = df
        agent._repairs_applied.append({
            "type": "outliers", "strategy": strategy, "columns": columns,
        })
        return ToolResult(status=ToolResultStatus.SUCCESS, output={
            "strategy": strategy, "columns_repaired": len(columns), "changes": changes,
        })

    except Exception as e:
        return ToolResult(status=ToolResultStatus.ERROR, error=f"Error repairing outliers: {str(e)}")
