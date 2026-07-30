"""Deterministic data-understanding helpers for the Streamlit studio.

The functions in this module calculate summaries and chart-ready frames.  They
do not call a model, mutate the uploaded frame, or decide whether a causal
assumption is true.  Keeping those boundaries here makes the interactive EDA
safe to cache and straightforward to test.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .profile import profile


ID_NAME = re.compile(r"^(unnamed.*|.*_?id|id_?.*|index|row|key|uuid)$", re.I)


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one compact, non-semantic summary row per column."""
    p = profile(df)
    rows: list[dict[str, Any]] = []
    for item in p.columns:
        unique_fraction = item.n_unique / max(1, p.n_rows)
        signals: list[str] = []
        if item.n_unique <= 1:
            signals.append("constant")
        if ID_NAME.match(item.name.strip()) or (
            item.numeric and unique_fraction > 0.95 and not item.binary
        ):
            signals.append("possible identifier")
        if item.binary:
            signals.append("binary")
        if item.datelike:
            signals.append("date candidate")
        rows.append({
            "column": item.name,
            "dtype": item.dtype,
            "missing %": round(item.missing * 100, 2),
            "distinct": item.n_unique,
            "unique %": round(unique_fraction * 100, 2),
            "minimum": item.low,
            "maximum": item.high,
            "signals": ", ".join(signals) or "—",
        })
    return pd.DataFrame(rows)


def chart_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Mechanical column groups used to populate EDA controls."""
    p = profile(df)
    categorical = [
        item.name
        for item in p.columns
        if 1 < item.n_unique <= 30 and (not item.numeric or item.binary)
    ]
    return {
        "all": p.names(),
        "numeric": p.numeric_names(),
        "categorical": categorical,
        "date": p.datelike_names(),
    }


def chart_sample(
    df: pd.DataFrame, columns: list[str] | tuple[str, ...], limit: int = 5_000
) -> pd.DataFrame:
    """Return a stable sample containing only the requested chart columns."""
    selected = [column for column in dict.fromkeys(columns) if column in df.columns]
    frame = df.loc[:, selected].copy()
    if len(frame) > limit:
        frame = frame.sample(n=limit, random_state=0).sort_index()
    return frame


def apply_cohort(df: pd.DataFrame, spec: dict[str, Any] | None) -> pd.DataFrame:
    """Apply one explicit, reviewable cohort filter to a copy of the data."""
    if not spec:
        return df.copy(deep=True)
    column = str(spec.get("column", ""))
    if column not in df.columns:
        raise ValueError(f"Cohort filter names missing column {column!r}")
    kind = spec.get("kind")
    if kind == "numeric_range":
        low, high = float(spec["low"]), float(spec["high"])
        numeric = pd.to_numeric(df[column], errors="coerce")
        return df.loc[numeric.between(low, high, inclusive="both")].copy()
    if kind == "categories":
        values = {str(value) for value in spec.get("values", [])}
        return df.loc[df[column].astype("string").isin(values)].copy()
    raise ValueError(f"Unsupported cohort filter kind {kind!r}")


def standardized_differences(
    df: pd.DataFrame, treatment: str, covariates: list[str] | tuple[str, ...]
) -> pd.DataFrame:
    """Compute unadjusted absolute standardised differences for binary arms."""
    if treatment not in df.columns:
        return pd.DataFrame(columns=["covariate", "standardised difference"])
    treatment_values = pd.to_numeric(df[treatment], errors="coerce")
    levels = sorted(treatment_values.dropna().unique())
    if len(levels) != 2:
        return pd.DataFrame(columns=["covariate", "standardised difference"])
    rows = []
    for column in covariates:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        left = values[treatment_values == levels[0]].dropna()
        right = values[treatment_values == levels[1]].dropna()
        if len(left) < 2 or len(right) < 2:
            continue
        pooled = np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2)
        difference = abs(float(right.mean() - left.mean())) / float(pooled) if pooled else 0.0
        rows.append({"covariate": column, "standardised difference": difference})
    if not rows:
        return pd.DataFrame(columns=["covariate", "standardised difference"])
    return pd.DataFrame(rows).sort_values("standardised difference", ascending=True)


def kaplan_meier_frame(
    df: pd.DataFrame, duration: str, event: str, treatment: str
) -> pd.DataFrame:
    """Return chart-ready Kaplan–Meier steps for each treatment group."""
    required = {duration, event, treatment}
    if not required <= set(df.columns):
        return pd.DataFrame(columns=["time", "survival", "group"])
    work = pd.DataFrame({
        "time": pd.to_numeric(df[duration], errors="coerce"),
        "event": pd.to_numeric(df[event], errors="coerce"),
        "group": df[treatment].astype("string"),
    }).dropna()
    curves: list[dict[str, Any]] = []
    for group, part in work.groupby("group", sort=True):
        part = part.sort_values("time")
        survival = 1.0
        curves.append({"time": 0.0, "survival": survival, "group": str(group)})
        for time_value in sorted(part.loc[part.event == 1, "time"].unique()):
            at_risk = int((part.time >= time_value).sum())
            events = int(((part.time == time_value) & (part.event == 1)).sum())
            if at_risk:
                survival *= 1 - events / at_risk
                curves.append({
                    "time": float(time_value),
                    "survival": float(survival),
                    "group": str(group),
                })
    return pd.DataFrame(curves)


def grouped_trends(
    df: pd.DataFrame, period: str, group: str, outcome: str
) -> pd.DataFrame:
    """Return mean outcomes by period and group for DiD inspection."""
    required = {period, group, outcome}
    if not required <= set(df.columns):
        return pd.DataFrame(columns=[period, group, "mean outcome", "rows"])
    work = df.loc[:, [period, group, outcome]].copy()
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    return (
        work.dropna()
        .groupby([period, group], as_index=False, dropna=False)[outcome]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean outcome", "count": "rows"})
    )
