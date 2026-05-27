"""get_data_overview - dataset shape, variable types, treatment/outcome summary."""

import numpy as np

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "get_data_overview",
    "description": "Get overview of the dataset including dimensions, variable types, missing data summary. Call this early to understand the data.",
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

    df = agent._df
    n_rows, n_cols = df.shape

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    binary_cols = [c for c in numeric_cols if df[c].dropna().nunique() == 2]

    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    total_missing_pct = df.isnull().sum().sum() / (n_rows * n_cols) * 100

    treatment_info = None
    if agent._treatment_var and agent._treatment_var in df.columns:
        t_vals = df[agent._treatment_var].value_counts().to_dict()
        treatment_info = {
            "variable": agent._treatment_var,
            "values": {str(k): int(v) for k, v in t_vals.items()},
            "missing": int(df[agent._treatment_var].isnull().sum()),
        }

    outcome_info = None
    if agent._outcome_var and agent._outcome_var in df.columns:
        outcome_info = {
            "variable": agent._outcome_var,
            "mean": round(float(df[agent._outcome_var].mean()), 3),
            "std": round(float(df[agent._outcome_var].std()), 3),
            "missing": int(df[agent._outcome_var].isnull().sum()),
        }

    confounders = []
    if state.data_profile and state.data_profile.potential_confounders:
        confounders = state.data_profile.potential_confounders[:10]

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "n_samples": n_rows,
            "n_features": n_cols,
            "numeric_columns": numeric_cols[:10],
            "categorical_columns": categorical_cols[:5],
            "binary_columns": binary_cols[:5],
            "total_missing_pct": round(total_missing_pct, 2),
            "columns_with_missing": len(cols_with_missing),
            "treatment": treatment_info,
            "outcome": outcome_info,
            "potential_confounders": confounders,
        },
    )
