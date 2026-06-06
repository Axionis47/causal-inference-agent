"""Pure helpers for the data repair agent."""

from pathlib import Path
from typing import Any

import pandas as pd


def load_dataframe(dataframe_path: str | None, logger) -> pd.DataFrame | None:
    """Read the parquet DataFrame the data_profiler wrote, or return None."""
    if dataframe_path and Path(dataframe_path).exists():
        try:
            return pd.read_parquet(dataframe_path)
        except Exception as e:
            logger.error("load_failed", error=str(e))
    return None


def save_dataframe(df: pd.DataFrame, dataframe_path: str | None, logger) -> None:
    """Overwrite the parquet DataFrame in place; failures are non-fatal."""
    if dataframe_path:
        try:
            df.to_parquet(dataframe_path)
        except Exception as e:
            logger.error("save_failed", error=str(e))


def count_outliers(data: pd.Series) -> int:
    """IQR-based outlier count for one numeric Series."""
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    return int(((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum())


def auto_finalize(
    df: pd.DataFrame,
    df_original: pd.DataFrame,
    repairs_applied: list[dict],
) -> dict[str, Any]:
    """Heuristic fallback when the agent never calls finalize_repairs."""
    repairs_summary = [
        f"{r['type']}: {r.get('strategy', '')} on {len(r.get('columns', []))} columns"
        for r in repairs_applied
    ]

    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    if missing_pct < 1:
        quality = "Good - minimal missing data"
    elif missing_pct < 5:
        quality = "Acceptable - some missing data remains"
    else:
        quality = "Fair - notable missing data remains"

    cautions: list[str] = []
    if len(df) < len(df_original) * 0.8:
        cautions.append("Significant row loss - check for selection bias")
    if len(df.columns) < len(df_original.columns) * 0.7:
        cautions.append("Many columns dropped - verify important covariates retained")

    return {
        "quality_assessment": quality,
        "repairs_summary": repairs_summary if repairs_summary else ["No repairs applied"],
        "cautions": cautions,
    }


def initial_observation_text(state, shape) -> str:
    """Frame the dataset + workflow for the repair agent without dumping rows."""
    dataset_name = state.dataset_info.name if state.dataset_info else "unknown"
    return f"""DATA REPAIR TASK
Dataset: {dataset_name}
Treatment variable: {state.treatment_variable}
Outcome variable: {state.outcome_variable}
Shape: {shape}

Your goal is to prepare this data for causal effect estimation while preserving
the causal structure and avoiding introduction of bias.

Start by calling get_data_summary to understand the current state, then
systematically:
1. Check for missing values and their patterns
2. Check for outliers in key variables
3. Check for collinearity among covariates
4. Apply repairs as needed, validating each one
5. Call finalize_repairs when data quality is acceptable

Be conservative - only repair issues that would materially affect causal inference."""
