"""get_data_summary - dataset shape, missing, treatment/outcome summary."""

import numpy as np

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "get_data_summary",
    "description": "Get summary of current dataset state including dimensions, missing counts, and treatment/outcome info. Call this FIRST.",
    "parameters": {},
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    df = agent._df
    n_rows, n_cols = df.shape

    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0].to_dict()
    total_missing_pct = float(df.isnull().sum().sum() / (n_rows * n_cols) * 100)

    treatment_col = agent._current_state.treatment_variable
    outcome_col = agent._current_state.outcome_variable

    treatment_info: dict = {"status": "Not found"}
    if treatment_col and treatment_col in df.columns:
        t_missing = int(df[treatment_col].isnull().sum())
        t_values = df[treatment_col].value_counts().to_dict()
        treatment_info = {
            "missing": t_missing,
            "values": {str(k): int(v) for k, v in t_values.items()},
        }

    outcome_info: dict = {"status": "Not found"}
    if outcome_col and outcome_col in df.columns:
        o_missing = int(df[outcome_col].isnull().sum())
        outcome_info = {
            "missing": o_missing,
            "mean": float(df[outcome_col].mean()),
            "std": float(df[outcome_col].std()),
        }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "shape": [n_rows, n_cols],
            "n_numeric_columns": len(numeric_cols),
            "total_missing_pct": total_missing_pct,
            "cols_with_missing": len(cols_with_missing),
            "treatment_variable": treatment_col,
            "treatment_info": treatment_info,
            "outcome_variable": outcome_col,
            "outcome_info": outcome_info,
            "repairs_applied": len(agent._repairs_applied),
        },
    )
