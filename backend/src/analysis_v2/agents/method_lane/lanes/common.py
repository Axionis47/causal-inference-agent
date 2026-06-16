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


def to_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric, tolerating thousands separators that survive CSV
    import ('1,234' -> 1234). Already-numeric columns pass through unchanged;
    anything still unparseable becomes NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    cleaned = series.astype("string").str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def numeric_frame_issues(
    frame: pd.DataFrame, columns: list[str], lane: str
) -> tuple[list[str], pd.DataFrame | None]:
    """Non-raising form of numeric_frame: (blocking reasons, complete-case view).
    The frame is returned even when the row count is short so callers can run
    further checks; it is None only when columns are missing outright."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        return [f"{lane}: columns missing from the dataset: {missing}"], None
    out = frame[columns].apply(to_numeric).dropna()
    if len(out) < MIN_COMPLETE_ROWS:
        return [f"{lane}: only {len(out)} complete numeric rows across {columns}"], out
    return [], out


def encode_design_issues(
    frame: pd.DataFrame, core: list[str], covariates: list[str], lane: str
) -> tuple[list[str], pd.DataFrame | None, list[str]]:
    """Non-raising design-matrix builder. `core` columns must be numeric;
    categorical `covariates` are one-hot encoded (kept as confounders, never
    silently dropped). Returns (issues, complete-case data, encoded covariate
    names). The conceptual covariate names stay with the caller; only the
    design matrix uses the expanded columns."""
    missing = [c for c in [*core, *covariates] if c not in frame.columns]
    if missing:
        return [f"{lane}: columns missing from the dataset: {missing}"], None, []
    parts: list[pd.DataFrame | pd.Series] = [frame[core].apply(to_numeric)]
    names: list[str] = []
    for c in covariates:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s) or to_numeric(s).notna().mean() > 0.5:
            parts.append(to_numeric(s).rename(c))
            names.append(c)
        else:
            dummies = pd.get_dummies(
                s.astype("object"), prefix=c, drop_first=True, dtype=float
            )
            parts.append(dummies)
            names.extend(dummies.columns.tolist())
    data = pd.concat(parts, axis=1).dropna()
    if len(data) < MIN_COMPLETE_ROWS:
        return [f"{lane}: only {len(data)} complete rows across the design"], data, names
    return [], data, names


def encode_design(
    frame: pd.DataFrame, core: list[str], covariates: list[str], lane: str
) -> tuple[pd.DataFrame, list[str]]:
    """Raising form of encode_design_issues."""
    issues, data, names = encode_design_issues(frame, core, covariates, lane)
    if issues:
        raise LaneInputError(issues[0])
    assert data is not None  # no issues implies a frame
    return data, names


def usable_covariates(
    frame: pd.DataFrame, covariates: list[str], threshold: float = 0.3
) -> tuple[list[str], list[str]]:
    """Covariates present and not mostly-missing. A column missing in more than
    `threshold` of rows would shrink the complete-case sample far more than it
    helps adjustment, so it is dropped here. Returns (kept, dropped)."""
    kept = [
        c for c in covariates
        if c in frame.columns and float(frame[c].isna().mean()) <= threshold
    ]
    dropped = [c for c in covariates if c not in kept]
    return kept, dropped


def drop_collinear(
    df: pd.DataFrame, protected: pd.DataFrame | None = None, rtol: float = 1e-8
) -> tuple[pd.DataFrame, list[str]]:
    """Drop covariate columns that are (near-)linear combinations of an intercept,
    the `protected` columns (the treatment and instrument, which must stay), and
    earlier-kept covariates, so the design is well conditioned. Two offenders:
    a complete dummy set or a sum-of-others column (south = southern regions),
    and a covariate collinear with the treatment itself (age = educ + exper + 6
    in the schooling data). Left in, either makes the fit singular and standard
    errors blow up. Rank uses singular values relative to the largest, so
    near-dependence is caught, not only exact. Returns (kept, dropped names)."""
    if df.shape[1] == 0:
        return df, []

    def _rank(mat: np.ndarray) -> int:
        s = np.linalg.svd(mat, compute_uv=False)
        return int((s > rtol * s[0]).sum()) if s.size else 0

    base = np.ones((len(df), 1))
    if protected is not None and protected.shape[1] > 0:
        base = np.column_stack([base, protected.to_numpy(dtype=float)])
    keep: list[str] = []
    dropped: list[str] = []
    rank = _rank(base)
    for col in df.columns:
        trial = np.column_stack([base, df[col].to_numpy(dtype=float)])
        trial_rank = _rank(trial)
        if trial_rank > rank:
            keep.append(col)
            base = trial
            rank = trial_rank
        else:
            dropped.append(col)
    return df[keep], dropped


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
