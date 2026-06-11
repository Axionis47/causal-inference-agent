"""Deterministic profiling tools: semantic typing and quality findings.

Pure functions over a DataFrame; no agent state, no LLM. Thresholds match
the input slice's gate profiler where they overlap (binary <=2 numeric
levels, ordinal <=10, categorical <=20 non-numeric levels) and add the
analysis-side findings: id-like columns, constants, duplicates,
high-missingness flags.
"""
from __future__ import annotations

import pandas as pd
from pandas.api import types as ptypes

from src.analysis_v2.spec import ColumnProfile, ProfileSummary

HIGH_MISSINGNESS = 0.20  # soft-warning threshold; hard failures live in gates

_ID_NAME_HINTS = ("id", "uuid", "guid", "index", "rownames", "unnamed: 0")
_TIME_NAME_HINTS = ("time", "date", "year", "month", "period", "day", "week")


def _looks_id_like(name: str, series: pd.Series, n_rows: int) -> bool:
    lowered = name.strip().lower()
    name_hit = lowered in _ID_NAME_HINTS or lowered.endswith("_id") or lowered.startswith(
        "unnamed"
    )
    if n_rows == 0:
        return name_hit
    nunique = series.nunique(dropna=True)
    all_unique = nunique == series.notna().sum() and nunique > 0
    if name_hit and all_unique:
        return True
    # A monotonically increasing unique integer column is a row index.
    if ptypes.is_integer_dtype(series.dtype) and all_unique and series.is_monotonic_increasing:
        return True
    return name_hit and nunique >= 0.95 * n_rows


def semantic_type(name: str, series: pd.Series, n_rows: int) -> str:
    if _looks_id_like(name, series, n_rows):
        return "id_like"
    if ptypes.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    nunique = series.nunique(dropna=True)
    if ptypes.is_numeric_dtype(series.dtype):
        if nunique <= 2:
            return "binary"
        if nunique <= 10:
            return "ordinal"
        return "numeric"
    if nunique <= 2:
        return "binary"
    if nunique <= 20:
        return "categorical"
    return "text"


def likely_time_columns(df: pd.DataFrame) -> list[str]:
    hits = []
    for name in df.columns:
        if ptypes.is_datetime64_any_dtype(df[name].dtype):
            hits.append(name)
            continue
        lowered = name.strip().lower()
        if any(h in lowered for h in _TIME_NAME_HINTS):
            hits.append(name)
    return hits


def build_profile_summary(df: pd.DataFrame) -> ProfileSummary:
    n_rows = len(df)
    columns: list[ColumnProfile] = []
    constant, id_like, high_missing = [], [], []

    for name in df.columns:
        series = df[name]
        missing = int(series.isna().sum())
        fraction = missing / n_rows if n_rows else 0.0
        stype = semantic_type(str(name), series, n_rows)
        nunique = int(series.nunique(dropna=True))
        is_constant = nunique <= 1
        sample = [str(v) for v in series.dropna().unique()[:5]]
        columns.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                semantic_type=stype,
                missing_count=missing,
                missing_fraction=round(fraction, 4),
                n_unique=nunique,
                is_constant=is_constant,
                sample_values=sample,
            )
        )
        if is_constant:
            constant.append(str(name))
        if stype == "id_like":
            id_like.append(str(name))
        if fraction >= HIGH_MISSINGNESS:
            high_missing.append(str(name))

    duplicates = int(df.duplicated().sum()) if n_rows else 0
    warnings = []
    if duplicates:
        warnings.append(f"{duplicates} duplicate rows ({duplicates / n_rows:.1%})")
    if constant:
        warnings.append(f"constant columns: {', '.join(constant)}")
    if high_missing:
        warnings.append(
            f"columns over {HIGH_MISSINGNESS:.0%} missing: {', '.join(high_missing)}"
        )

    return ProfileSummary(
        n_rows=n_rows,
        n_columns=len(df.columns),
        columns=columns,
        duplicate_row_count=duplicates,
        constant_columns=constant,
        id_like_columns=id_like,
        high_missingness_columns=high_missing,
        likely_time_columns=likely_time_columns(df),
        warnings=warnings,
    )


def column_inventory_frame(summary: ProfileSummary) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": c.name,
                "dtype": c.dtype,
                "semantic_type": c.semantic_type,
                "missing_fraction": c.missing_fraction,
                "n_unique": c.n_unique,
                "is_constant": c.is_constant,
            }
            for c in summary.columns
        ]
    )
