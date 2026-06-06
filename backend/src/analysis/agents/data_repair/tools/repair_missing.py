"""repair_missing - imputation or row/column dropping for missing values."""

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "repair_missing",
    "description": "Repair missing values in specified columns using chosen strategy.",
    "parameters": {
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns to repair (empty for all with missing)",
        },
        "strategy": {
            "type": "string",
            "enum": ["mean", "median", "mode", "iterative", "drop_rows", "drop_columns"],
            "description": "Imputation strategy",
        },
        "drop_threshold": {
            "type": "number",
            "description": "For drop strategies: missing rate threshold (0-1)",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    strategy: str = "median",
    columns: list[str] | None = None,
    drop_threshold: float = 0.5,
    **kwargs,
) -> ToolResult:
    df = agent._df

    if not columns:
        columns = df.columns[df.isnull().any()].tolist()

    protected = [agent._current_state.treatment_variable, agent._current_state.outcome_variable]
    if strategy in ["drop_columns"]:
        columns = [c for c in columns if c not in protected]

    columns = [c for c in columns if c in df.columns]

    if not columns:
        return ToolResult(status=ToolResultStatus.SUCCESS, output={"message": "No columns to repair."})

    before_missing = int(df[columns].isnull().sum().sum())

    try:
        if strategy == "drop_rows":
            rows_before = len(df)
            df = df.dropna(subset=columns, thresh=int(len(columns) * (1 - drop_threshold)))
            agent._df = df
            rows_dropped = rows_before - len(df)
            agent._repairs_applied.append({
                "type": "missing", "strategy": "drop_rows", "columns": columns,
                "rows_dropped": rows_dropped,
            })
            return ToolResult(status=ToolResultStatus.SUCCESS, output={
                "action": "drop_rows", "rows_dropped": rows_dropped, "new_shape": list(df.shape),
            })

        elif strategy == "drop_columns":
            cols_to_drop = [col for col in columns if df[col].isnull().mean() > drop_threshold]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                agent._df = df
                agent._repairs_applied.append({
                    "type": "missing", "strategy": "drop_columns", "columns": cols_to_drop,
                })
                return ToolResult(status=ToolResultStatus.SUCCESS, output={
                    "action": "drop_columns", "columns_dropped": cols_to_drop,
                })
            return ToolResult(status=ToolResultStatus.SUCCESS, output={"message": "No columns exceeded threshold."})

        elif strategy == "iterative":
            numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                imputer = IterativeImputer(max_iter=10, random_state=42)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                agent._df = df

        elif strategy in ["mean", "median"]:
            numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_cols:
                imputer = SimpleImputer(strategy=strategy)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                agent._df = df

        elif strategy == "mode":
            for col in columns:
                if df[col].isnull().any():
                    mode_val = df[col].mode()
                    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
                    df[col].fillna(fill_val, inplace=True)
            agent._df = df

        after_missing = int(df[columns].isnull().sum().sum())
        agent._repairs_applied.append({
            "type": "missing", "strategy": strategy, "columns": columns,
            "before": before_missing, "after": after_missing,
        })
        return ToolResult(status=ToolResultStatus.SUCCESS, output={
            "strategy": strategy, "before_missing": before_missing, "after_missing": after_missing,
        })

    except Exception as e:
        return ToolResult(status=ToolResultStatus.ERROR, error=f"Error repairing missing values: {str(e)}")
