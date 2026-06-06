"""validate_current_state - composite data-quality score for the current DataFrame."""

import numpy as np

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from ..helpers import count_outliers

SCHEMA = {
    "name": "validate_current_state",
    "description": "Validate current data quality and check if more repairs are needed.",
    "parameters": {},
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    df = agent._df
    df_orig = agent._df_original

    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    row_loss = (len(df_orig) - len(df)) / len(df_orig) * 100
    col_loss = (len(df_orig.columns) - len(df.columns)) / len(df_orig.columns) * 100

    treatment_ok = agent._current_state.treatment_variable in df.columns
    outcome_ok = agent._current_state.outcome_variable in df.columns

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    total_outliers = 0
    total_values = 0
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) >= 10:
            total_outliers += count_outliers(data)
            total_values += len(data)
    outlier_pct = total_outliers / total_values * 100 if total_values > 0 else 0

    score = 100
    score -= missing_pct * 2
    score -= min(20, row_loss)
    score -= min(10, col_loss)
    score -= min(20, outlier_pct * 2)
    score = max(0, score)

    if score >= 80:
        status_msg = "GOOD - Data is ready for causal analysis."
    elif score >= 60:
        status_msg = "ACCEPTABLE - Minor issues remain but analysis can proceed."
    else:
        status_msg = "NEEDS_WORK - Consider additional repairs."

    critical_warning = None
    if not treatment_ok or not outcome_ok:
        critical_warning = "Treatment or outcome variable is missing! Repairs may have been too aggressive."

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "shape": list(df.shape),
            "original_shape": list(df_orig.shape),
            "row_loss_pct": round(row_loss, 1),
            "column_loss_pct": round(col_loss, 1),
            "missing_data_pct": round(missing_pct, 2),
            "outlier_rate_pct": round(outlier_pct, 1),
            "treatment_variable_present": treatment_ok,
            "outcome_variable_present": outcome_ok,
            "quality_score": round(score),
            "status": status_msg,
            "critical_warning": critical_warning,
        },
    )
