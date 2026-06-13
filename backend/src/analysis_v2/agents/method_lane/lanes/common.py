"""Shared shapes for method lanes.

A lane is a function (frame, plan, spec) -> LaneOutcome. It runs exactly
one estimator family, returns the typed EstimateResult plus renderable
artifacts, and raises LaneInputError when its required inputs are
missing or degenerate (treatment without variation, empty cells). It
never switches design.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analysis_v2.spec import EstimateResult


class LaneInputError(ValueError):
    """The lane cannot run on this data; the message says exactly why."""


@dataclass
class LaneArtifact:
    name: str  # artifact id suffix and file stem
    kind: str  # "markdown" | "table" | "plot"
    title: str
    payload: object  # str for markdown, DataFrame for table, bytes for plot


@dataclass
class LaneOutcome:
    result: EstimateResult
    artifacts: list[LaneArtifact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


MIN_COMPLETE_ROWS = 20

# Each check below has a non-raising "issue" form (returns a list of blocking
# reasons, empty when ready) and a raising form. The lane's run() raises; the
# readiness checker collects. Both call the same issue function, so the gate
# and the lane can never disagree about what makes the data runnable.


def numeric_frame_issues(
    frame: pd.DataFrame, columns: list[str], lane: str
) -> tuple[list[str], pd.DataFrame | None]:
    """Non-raising form of numeric_frame: (blocking reasons, complete-case view).
    The frame is returned even when the row count is short so callers can run
    further checks; it is None only when columns are missing outright."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        return [f"{lane}: columns missing from the dataset: {missing}"], None
    out = frame[columns].apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna()
    if len(out) < MIN_COMPLETE_ROWS:
        return [f"{lane}: only {len(out)} complete numeric rows across {columns}"], out
    return [], out


def numeric_frame(frame: pd.DataFrame, columns: list[str], lane: str) -> pd.DataFrame:
    """Complete-case numeric view of the named columns, validated."""
    issues, out = numeric_frame_issues(frame, columns, lane)
    if issues:
        raise LaneInputError(issues[0])
    assert out is not None  # no issues implies a frame
    return out


def variation_issue(series: pd.Series, label: str, lane: str) -> list[str]:
    return [] if series.nunique(dropna=True) >= 2 else [f"{lane}: {label} has no variation"]


def require_variation(series: pd.Series, label: str, lane: str) -> None:
    issues = variation_issue(series, label, lane)
    if issues:
        raise LaneInputError(issues[0])


def binary_issue(series: pd.Series, label: str, lane: str) -> list[str]:
    values = sorted(series.dropna().unique(), key=str)
    if len(values) != 2:
        return [f"{lane}: {label} is not binary ({len(values)} levels)"]
    return []


def binary_01(series: pd.Series, label: str, lane: str) -> pd.Series:
    """Map a two-valued series onto {0,1}, higher/True value = 1."""
    issues = binary_issue(series, label, lane)
    if issues:
        raise LaneInputError(issues[0])
    values = sorted(series.dropna().unique(), key=str)
    return (series == values[1]).astype(int) if not pd.api.types.is_numeric_dtype(
        series
    ) else (series == max(values)).astype(int)


def fmt(value: float) -> str:
    return f"{value:,.4g}"


def ci_from(coef: float, se: float) -> tuple[float, float]:
    return coef - 1.96 * se, coef + 1.96 * se


def summary_markdown(title: str, lines: list[str], model_text: str | None = None) -> str:
    body = f"## {title}\n\n" + "\n".join(f"- {line}" for line in lines)
    if model_text:
        body += "\n\n```\n" + model_text[:6000] + "\n```"
    return body


def effects_table(result: EstimateResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "estimand": e.estimand,
                "estimate": e.estimate,
                "std_error": e.std_error,
                "ci_lower": e.ci_lower,
                "ci_upper": e.ci_upper,
                "p_value": e.p_value,
            }
            for e in result.effects
        ]
    )


def safe_float(value) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise LaneInputError("estimator produced a non-finite value")
    return out
