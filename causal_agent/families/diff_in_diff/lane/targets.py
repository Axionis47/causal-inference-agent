"""Outcome-blind target masks and weights."""

from __future__ import annotations

import pandas as pd


def target_groups(panel: pd.DataFrame, target: dict) -> list[tuple[str, pd.Series]]:
    mode = target["aggregation"]
    treated = panel.treat == 1
    hs = target.get("horizons") or sorted(int(x) for x in panel.loc[treated, "rel_time"].unique() if x >= 0)
    gs = target.get("cohorts") or sorted(str(int(x)) for x in panel.loc[treated, "cohort"].unique())
    if mode == "pooled":
        return [("pooled", treated)]
    if mode in {"horizon", "horizon_profile"}:
        return [(f"horizon:{h}", treated & (panel.rel_time == int(h))) for h in hs]
    if mode in {"cohort", "cohort_profile"}:
        return [(f"cohort:{g}", treated & (panel.cohort == int(g))) for g in gs]
    return [(f"cohort:{g}:horizon:{h}", treated & (panel.cohort == int(g)) & (panel.rel_time == int(h))) for g in gs for h in hs]


def weights_of(panel: pd.DataFrame, target: dict) -> dict[str, float]:
    """Shares are fixed from row support, never effect size or significance."""
    rows = panel.loc[target_groups(panel, target)[0][1]]
    counts = rows.groupby(["cohort", "rel_time"]).size()
    total = int(counts.sum())
    return {f"{int(g)}:{int(h)}": int(n) / total for (g, h), n in counts.items()} if total else {}


def projection_weights(panel: pd.DataFrame, controls: list[str]) -> dict[str, float]:
    """The static coefficient's treated-row weights, computed without reading outcomes."""
    import numpy as np

    fixed = pd.get_dummies(panel[["unit", "time_index"]].astype(str), dtype=float)
    X = np.column_stack([np.ones(len(panel)), fixed.to_numpy(), *[panel[c].to_numpy() for c in controls]])
    D = panel.treat.to_numpy(dtype=float)
    residual = D - X @ np.linalg.lstsq(X, D, rcond=None)[0]
    denom = float(residual @ residual)
    if denom <= 1e-12:
        raise ValueError("treatment has no residual variation in the frozen design")
    return {str(int(rid)): float(w / denom) for rid, w, on in zip(panel.row_id, residual, D, strict=True) if on == 1}


def clean_projection_support(panel: pd.DataFrame, horizons: list[int]) -> tuple[list[str], dict[str, float]]:
    """Adoption observations and regression weights for the frozen clean-comparison window."""
    import numpy as np

    p = panel.sort_values(["unit", "time_index"]).copy()
    H = max(horizons)
    by = p.groupby("unit")
    p["enter"] = p.treat - by.treat.shift(1)
    complete = by.y.shift(1).notna()
    for h in range(H + 1):
        complete &= by.y.shift(-h).notna()
    clean = (p.enter == 1) | (by.treat.shift(-H) == 0)
    sample = p.loc[complete & clean].copy()
    if sample.empty or not (sample.enter == 1).any():
        raise ValueError("no complete clean projection comparisons")
    X = np.column_stack([np.ones(len(sample)), pd.get_dummies(sample.time_index.astype(str), dtype=float).to_numpy()])
    D = sample.enter.to_numpy(dtype=float)
    resid = D - X @ np.linalg.lstsq(X, D, rcond=None)[0]
    denom = float(resid @ resid)
    if denom <= 1e-12:
        raise ValueError("no within-period clean projection contrast")
    weights = {str(int(rowid)): float(w / denom) for rowid, w, on in zip(sample.row_id, resid, D, strict=True) if on == 1}
    return sorted(str(x) for x in sample.loc[sample.enter == 1, "unit"].unique()), weights


def horizon_projection_weights(panel: pd.DataFrame, controls: list[str], horizon: int) -> dict[str, float]:
    """The dynamic coefficient's exact linear score on outcomes, using no outcomes."""
    import numpy as np

    periods = sorted(int(x) for x in panel.loc[panel.treated == 1, "rel_time"].unique() if x != -1)
    if horizon not in periods:
        raise ValueError("requested horizon is absent")
    columns = {f"h{k}": ((panel.treated == 1) & (panel.rel_time == k)).astype(float) for k in periods}
    columns.update({c: panel[c].astype(float) for c in controls})
    centered = pd.DataFrame(columns, index=panel.index)
    for _ in range(200):
        before = centered.to_numpy().copy()
        centered -= centered.groupby(panel.unit).transform("mean")
        centered -= centered.groupby(panel.time_index).transform("mean")
        if np.max(np.abs(centered.to_numpy() - before)) < 1e-11:
            break
    else:
        raise ValueError("fixed-effect projection did not converge")
    D = centered[f"h{horizon}"].to_numpy()
    others = centered.drop(columns=f"h{horizon}").to_numpy()
    residual = D - others @ np.linalg.lstsq(others, D, rcond=None)[0] if others.shape[1] else D
    denom = float(residual @ residual)
    if denom <= 1e-12:
        raise ValueError("the requested horizon is collinear")
    return {str(int(rowid)): float(score / denom) for rowid, score in zip(panel.row_id, residual, strict=True) if abs(score) > 1e-12}
