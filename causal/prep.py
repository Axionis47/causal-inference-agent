"""Shared data preparation. Every lane starts here.

These are the only assumptions the lanes make about their input: the columns
exist, they are numeric, complete rows remain, and treatment varies. Anything
a lane needs beyond that, it checks itself.
"""
from __future__ import annotations

import pandas as pd

from .estimate import LaneError

MIN_ROWS = 20  # below this, no estimator is worth reporting


def require_columns(df: pd.DataFrame, columns: list[str], lane: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise LaneError(f"{lane}: dataset has no column(s) {missing}")


def numeric_frame(df: pd.DataFrame, columns: list[str], lane: str) -> pd.DataFrame:
    """The named columns, coerced to numbers, complete rows only.

    Strings that look like numbers ("1,200") are coerced; anything else becomes
    NaN and its row is dropped, so a categorical column silently costs you rows.
    That is why callers pass only columns they intend to be numeric.
    """
    require_columns(df, columns, lane)

    def to_float(s: pd.Series) -> pd.Series:
        if not pd.api.types.is_numeric_dtype(s):
            s = s.astype(str).str.replace(",", "", regex=False)
        # float throughout: statsmodels cannot build a design matrix from bool
        return pd.to_numeric(s, errors="coerce").astype(float)

    out = pd.DataFrame({c: to_float(df[c]) for c in columns}).dropna()
    if len(out) < MIN_ROWS:
        raise LaneError(
            f"{lane}: only {len(out)} complete numeric rows across {columns} "
            f"(need {MIN_ROWS})"
        )
    return out


def require_variation(series: pd.Series, label: str, lane: str) -> None:
    if series.nunique(dropna=True) < 2:
        raise LaneError(f"{lane}: {label} takes only one value")


def as_binary(series: pd.Series, label: str, lane: str) -> pd.Series:
    """Map a two-valued column to 0/1, the higher value becoming 1."""
    levels = sorted(series.dropna().unique())
    if len(levels) != 2:
        raise LaneError(
            f"{lane}: {label} must have exactly 2 values, found {len(levels)}"
        )
    return (series == levels[1]).astype(int)


def ci95(value: float, se: float | None) -> tuple[float | None, float | None]:
    """The 95% normal interval, or (None, None) when there is no usable SE."""
    if se is None or not pd.notna(se):
        return None, None
    return value - 1.96 * se, value + 1.96 * se
