"""What the data looks like, computed without asking a model anything.

The profile is deliberately dumb: it reports shape and type, and guesses a role
only where the guess is mechanical (a 0/1 column is binary; a column that parses
as dates is a date). Deciding what a column *means* is the intake step's job,
and deciding what design it supports is design.py's.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Column:
    name: str
    dtype: str
    missing: float  # fraction, 0.0 to 1.0
    n_unique: int
    numeric: bool
    binary: bool
    datelike: bool
    low: float | None = None
    high: float | None = None

    def describe(self) -> str:
        bits = [self.dtype, f"{self.n_unique} distinct"]
        if self.missing:
            bits.append(f"{self.missing:.0%} missing")
        if self.binary:
            bits.append("binary")
        if self.datelike:
            bits.append("dates")
        if self.low is not None:
            bits.append(f"range {self.low:.4g} to {self.high:.4g}")
        return f"{self.name}: " + ", ".join(bits)


@dataclass
class Profile:
    n_rows: int
    columns: list[Column]

    def __getitem__(self, name: str) -> Column:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def numeric_names(self) -> list[str]:
        return [c.name for c in self.columns if c.numeric]

    def binary_names(self) -> list[str]:
        return [c.name for c in self.columns if c.binary]

    def datelike_names(self) -> list[str]:
        return [c.name for c in self.columns if c.datelike]

    def as_text(self) -> str:
        """The column list, for showing a person or putting in a prompt."""
        return "\n".join(c.describe() for c in self.columns)


def _as_numeric(series: pd.Series) -> pd.Series | None:
    """The column as numbers, or None if it isn't one.

    Mirrors what prep.numeric_frame will accept, thousands separators included,
    so the profile never claims a column is unusable that the lanes can read.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    notna = series.notna()
    if notna.sum() and bool(parsed[notna].notna().mean() > 0.9):
        return parsed
    return None


def _is_datelike(series: pd.Series, numeric: bool) -> bool:
    """True when most non-null values parse as dates.

    Numbers are never dates. Without that guard a plain integer parses as a
    year, and "1,234" parses as a date, so a visit count would be reported as
    a timestamp.
    """
    if numeric:
        return False
    sample = series.dropna().head(200)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    # bool(), not numpy.bool_: the graph checkpointer serialises this state
    # with msgpack, which refuses numpy scalars.
    return bool(parsed.notna().mean() > 0.9)


def profile(df: pd.DataFrame) -> Profile:
    columns = []
    for name in df.columns:
        s = df[name]
        as_num = _as_numeric(s)
        numeric = as_num is not None
        values = s.dropna()
        low = high = None
        if numeric:
            nums = as_num.dropna()
            if not nums.empty:
                low, high = float(nums.min()), float(nums.max())
        columns.append(
            Column(
                name=str(name),
                dtype=str(s.dtype),
                missing=float(s.isna().mean()),
                n_unique=int(values.nunique()),
                numeric=bool(numeric),
                binary=bool(values.nunique() == 2),
                datelike=_is_datelike(s, numeric),
                low=low,
                high=high,
            )
        )
    return Profile(n_rows=len(df), columns=columns)
