"""Permitted pre-freeze diagnostics. No outcome after first adoption reaches a fit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def regular_support(panel: pd.DataFrame) -> bool:
    """Rows correspond to consecutive declared periods within every retained unit."""
    for _, rows in panel.sort_values(["unit", "time_index"]).groupby("unit"):
        idx = rows.time_index.to_numpy()
        if len(idx) > 1 and not np.all(np.diff(idx) == 1):
            return False
    vals = sorted(panel.time.unique())
    if len(vals) < 3:
        return True
    try:
        gaps = np.diff(pd.to_numeric(pd.Series(vals)).to_numpy())
        return bool(np.allclose(gaps, gaps[0]))
    except (TypeError, ValueError):
        dt = pd.DatetimeIndex(vals)
        return pd.infer_freq(dt) is not None


def pre_trends(panel: pd.DataFrame, controls: list[str], cluster: str) -> dict[str, Any]:
    """Globally pre-earliest-adoption group diagnostic; passing never certifies parallel trends."""
    import pyfixest as pf

    pre = panel.loc[panel.post == 0].copy()
    times = sorted(pre.time_index.unique())
    base = dict(periods=len(times), paths=[], p=None, raw_p=None, coefficients={}, error=None)
    base["paths"] = pre.groupby(["time_index", "treated"]).y.agg(["mean", "count"]).reset_index().to_dict("records")
    if len(times) < 2 or pre.treated.nunique() < 2:
        base["error"] = "pretrend diagnostic untestable: fewer than two pre periods or one comparison arm"
        return base
    controls = [c for c in controls if c in pre and pre[c].nunique() > 1]
    formula = f"y ~ i(time_index, treated, ref={times[-1]})" + "".join(f" + {c}" for c in controls) + " | unit+time_index"
    try:
        m = pf.feols(formula, pre, vcov={"CRV1": cluster})
        names = [n for n in m.coef().index if n.startswith("time_index::")]
        if not names:
            base["error"] = "no estimable pre-period interactions"
            return base
        R = np.zeros((len(names), len(m.coef())))
        for i, n in enumerate(names):
            R[i, list(m.coef().index).index(n)] = 1
        test = m.wald_test(R=R, q=np.zeros(len(names)), distribution="chi2")
        base.update(p=float(test["pvalue"]), raw_p=float(test["pvalue"]), coefficients=m.coef()[names].to_dict())
    except Exception as ex:
        base["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    return base


def holm(pvalues: list[float | None]) -> list[float | None]:
    present = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    out: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, (p, i) in enumerate(present):
        running = max(running, min(1.0, p * (len(present) - rank)))
        out[i] = running
    return out


def untreated_support(panel: pd.DataFrame) -> bool:
    untreated = panel.loc[panel.treat == 0]
    return set(panel.unit) <= set(untreated.unit) and set(panel.time_index) <= set(untreated.time_index)
