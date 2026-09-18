"""From the raw table to the canonical table rdrobust's surface expects. Facts only.

Canonical columns: y, x, side, plus t when a take-up column exists, cluster when the dataset declares an
entity, and every numeric candidate covariate (NaN allowed; the library drops incomplete rows only when a
covariate enters a fit). x is the score recentred on the cutoff and flipped so the treated side is positive,
so every fit, check, and placebo downstream is written for one geometry with the cutoff at 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_agent.specialists.rd.contracts import Score, ShapeFacts

CANON = ["y", "x", "side"]


def covcol(key: str) -> str:
    """Where a raw covariate lives in the canonical table. Raw keys such as y, t, or x would otherwise collide."""
    return f"cov_{key}"


class ShapeError(Exception):
    """A typed stop: the table cannot be put into cutoff form. `facts` explain why."""

    def __init__(self, reason: str, facts: list[str], fix: str):
        super().__init__(reason)
        self.reason, self.facts, self.fix = reason, facts, fix


def effective_cutoff(x_raw: pd.Series, score: Score) -> tuple[float, float]:
    """The cutoff the fits use, and how far it moved.

    The library counts a score equal to the cutoff as treated. When the notes say the cutoff value is control
    (a strict rule), the effective cutoff moves to the midpoint between the cutoff and the next distinct score on
    the treated side, so no row sits at zero after recentring.
    """
    c = float(score.cutoff)
    finite = x_raw[np.isfinite(x_raw)]
    if score.cutoff_value_treated or not (finite == c).any():
        return c, 0.0
    vals = np.sort(finite.unique())
    if score.treated_side == "above":
        nxt = vals[vals > c]
        c_eff = (c + float(nxt.min())) / 2 if len(nxt) else c
    else:
        nxt = vals[vals < c]
        c_eff = (c + float(nxt.max())) / 2 if len(nxt) else c
    return c_eff, c_eff - c


def canonical(table: pd.DataFrame, score: Score, outcome: str, candidates: list[str], cluster_column: str | None, cfg: dict) -> tuple[pd.DataFrame, pd.Series, ShapeFacts]:
    """Returns the canonical table (primary rows), the recentred scores of every finite-score row in their recorded
    orientation (for the density test), and the shape facts. Raises ShapeError when a side is too thin."""
    x_raw = pd.to_numeric(table[score.column], errors="coerce")
    sign = 1.0 if score.treated_side == "above" else -1.0
    c = float(score.cutoff)
    finite = np.isfinite(x_raw)
    rows_at_cutoff = int((x_raw[finite] == c).sum())
    c_eff, shift = effective_cutoff(x_raw, score)
    x_all = (x_raw - c_eff)[finite]  # recorded orientation: the density test is not symmetric under mass points

    df = pd.DataFrame({"y": pd.to_numeric(table[outcome], errors="coerce"), "x": sign * (x_raw - c_eff)})
    if score.takeup_column:
        col = table[score.takeup_column]
        t = (col.astype(str) == str(score.takeup_level)).astype(float)
        t[col.isna()] = np.nan
        df["t"] = t
    if cluster_column and cluster_column in table.columns:
        df["cluster"] = table[cluster_column].astype(str)
    numeric_candidates = [k for k in candidates if k in table.columns and pd.api.types.is_numeric_dtype(table[k])]
    for k in numeric_candidates:
        df[covcol(k)] = pd.to_numeric(table[k], errors="coerce")
    needed = ["y", "x"] + (["t"] if "t" in df.columns else [])
    df = df[np.isfinite(df[needed]).all(axis=1)].copy()
    df["side"] = (df["x"] >= 0).astype(int)

    facts = _facts(df, x_all, table, x_raw, numeric_candidates, rows_at_cutoff, shift, cluster_column, cfg)
    _guard(facts, cfg)
    order = CANON + (["t"] if "t" in df.columns else []) + (["cluster"] if "cluster" in df.columns else []) + [covcol(k) for k in numeric_candidates]
    return df[order].reset_index(drop=True), x_all.reset_index(drop=True), facts


def rows_complete_on(df: pd.DataFrame, columns: list[str]) -> int:
    """columns are raw keys; they live under covcol(key) in the canonical table."""
    cols = [covcol(k) for k in columns if covcol(k) in df.columns]
    return int(np.isfinite(df[cols]).all(axis=1).sum()) if cols else int(len(df))


# ------------------------------------------------------------------ helpers


def _facts(df, x_all, table, x_raw, numeric_candidates, rows_at_cutoff, shift, cluster_column, cfg) -> ShapeFacts:
    left, right = df[df["side"] == 0], df[df["side"] == 1]
    tl = float(left["t"].mean()) if "t" in df.columns and len(left) else None
    tr = float(right["t"].mean()) if "t" in df.columns and len(right) else None
    rule = cfg["compliance"]["sharp_when"]
    if "t" not in df.columns:
        kind = "sharp"
    else:
        kind = "sharp" if (tr is not None and tl is not None and tr >= rule["treated_side_min"] and tl <= rule["other_side_max"]) else "fuzzy"
    finite = x_raw[np.isfinite(x_raw)]
    return ShapeFacts(
        rows_file=int(len(table)), rows_score=int(len(x_all)), rows_primary=int(len(df)), rows_covariates=rows_complete_on(df, numeric_candidates),
        n_left=int(len(left)), n_right=int(len(right)), takeup_left=tl, takeup_right=tr, kind=kind,
        distinct_scores=int(df["x"].nunique()),
        duplicate_share_left=float(1 - left["x"].nunique() / len(left)) if len(left) else 0.0,
        duplicate_share_right=float(1 - right["x"].nunique() / len(right)) if len(right) else 0.0,
        rows_at_cutoff=rows_at_cutoff, cutoff_shift=float(shift),
        score_min=float(finite.min()) if len(finite) else float("nan"), score_max=float(finite.max()) if len(finite) else float("nan"),
        cluster_column=cluster_column if cluster_column and "cluster" in df.columns else None,
    )


def _guard(f: ShapeFacts, cfg: dict) -> None:
    hard = int(cfg["sides"]["min_rows"]["hard"])
    if f.n_left < hard or f.n_right < hard:
        raise ShapeError("too few rows on one side of the cutoff", [f"{f.n_left} rows on the control side, {f.n_right} on the treated side; the floor is {hard} a side"],
                         "more units with scores on both sides of the cutoff")
