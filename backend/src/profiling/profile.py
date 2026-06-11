"""Deterministic structural profile of a dataset, computed before the gate.

Pure pandas/numpy, reproducible from the frame alone: column types, missingness,
per-column stats, and deterministic time detection. No LLM, no role inference.
Treatment and outcome come from the user, so no candidate lists are produced.
See docs/input-slice/confirmed-dataset-format.md section 7.
"""
from __future__ import annotations

import pandas as pd

from src.analysis_v2.state import DataProfile


def compute_basic_profile(df: pd.DataFrame) -> DataProfile:
    """Compute the shape, types, missing-count, and per-column stats of a frame."""
    if df is None or len(df) == 0:
        return DataProfile(
            n_samples=0,
            n_features=0 if df is None else len(df.columns),
            feature_names=[] if df is None else list(df.columns),
            feature_types={},
            missing_values={},
            numeric_stats={},
            categorical_stats={},
        )

    feature_types: dict[str, str] = {}
    numeric_stats: dict[str, dict[str, float]] = {}
    categorical_stats: dict[str, dict[str, int]] = {}
    missing_values: dict[str, int] = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_count = df[col].nunique()
            if unique_count <= 2:
                feature_types[col] = "binary"
            elif unique_count <= 10:
                feature_types[col] = "ordinal"
            else:
                feature_types[col] = "numeric"

            numeric_stats[col] = {
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else 0,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else 0,
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else 0,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else 0,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else 0,
            }
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            feature_types[col] = "datetime"
        else:
            if df[col].nunique() <= 20:
                feature_types[col] = "categorical"
                value_counts = df[col].value_counts().head(10).to_dict()
                categorical_stats[col] = {str(k): int(v) for k, v in value_counts.items()}
            else:
                feature_types[col] = "text"

        missing_values[col] = int(df[col].isna().sum())

    return DataProfile(
        n_samples=len(df),
        n_features=len(df.columns),
        feature_names=list(df.columns),
        feature_types=feature_types,
        missing_values=missing_values,
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
    )


_TIME_COLUMN_KEYWORDS = ["time", "date", "year", "month", "period"]


def detect_time_column(
    df: pd.DataFrame, feature_types: dict[str, str]
) -> tuple[bool, str | None]:
    """Deterministically pick a single time column, or none.

    A column qualifies if it is a datetime dtype, or its name contains a time
    keyword. The first match wins. Pure: no LLM, reproducible from the frame.
    """
    for col in df.columns:
        if feature_types.get(col) == "datetime":
            return True, col
        if any(kw in col.lower() for kw in _TIME_COLUMN_KEYWORDS):
            return True, col
    return False, None


def compute_deterministic_profile(df: pd.DataFrame) -> DataProfile:
    """The facts-only profile used before the data-review gate.

    The descriptive profile (types, stats, missingness) plus deterministic time
    detection. Leaves causal roles to the user; stores no machine guess.
    """
    profile = compute_basic_profile(df)
    profile.has_time_dimension, profile.time_column = detect_time_column(
        df, profile.feature_types
    )
    return profile
