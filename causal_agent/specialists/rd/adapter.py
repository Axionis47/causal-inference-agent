"""The only file that imports rdrobust and rddensity. Reads a spec and the canonical table, returns facts.

One fit function for every spec. Inference is a vce and a cluster passed through. Falsifications refit the
same function on a subset or with a fixed bandwidth. Failures come back as facts on the record, never as
exceptions. Every call runs with numpy errors silenced (the Mac BLAS build warns on every matmul), warnings
captured as notes, and stdout captured for the library's printed diagnostics.
"""

from __future__ import annotations

import contextlib
import io
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from scipy.stats import norm

from causal_agent.common.contracts import Estimate

POINT_ROW, INTERVAL_ROW = 0, 2  # Conventional, Robust


@dataclass
class Fit:
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p: float | None = None
    se: float | None = None
    h_left: float | None = None
    h_right: float | None = None
    b_left: float | None = None
    b_right: float | None = None
    n_left: int = 0
    n_right: int = 0
    n_h_left: int = 0
    n_h_right: int = 0
    vce: str = ""
    model: str = ""
    first_stage: dict | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def h(self) -> float | None:
        return self.h_left

    @property
    def b(self) -> float | None:
        return self.b_left

    def covers_zero(self) -> bool | None:
        if self.ci_low is None or self.ci_high is None:
            return None
        return self.ci_low <= 0 <= self.ci_high


def _run(fn, **kw):
    buf = io.StringIO()
    with np.errstate(all="ignore"), warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(buf):
        warnings.simplefilter("always")
        res = fn(**kw)
    notes = [str(w.message) for w in caught if not issubclass(w.category, RuntimeWarning)]
    notes += [line.strip() for line in buf.getvalue().splitlines() if line.strip()]
    return res, notes


def _kwargs(params: dict, df: pd.DataFrame, *, y: str, fuzzy: bool, covs: list[str] | None, cluster: bool, vce: str, c: float, bwselect: str | None) -> dict:
    kw: dict[str, Any] = dict(
        y=df[y],
        x=df["x"],
        c=c,
        p=int(params.get("p", 1)),
        kernel=params.get("kernel", "tri"),
        bwselect=bwselect or params.get("bwselect", "mserd"),
        vce=vce,
        masspoints=params.get("masspoints", "adjust"),
        level=int(params.get("level", 95)),
    )
    if fuzzy:
        kw["fuzzy"] = df["t"]
    if covs:
        kw["covs"] = df[list(covs)]
    if cluster and "cluster" in df.columns:
        kw["cluster"] = df["cluster"]
    return kw


def fit(
    params: dict,
    table: pd.DataFrame,
    *,
    y: str = "y",
    fuzzy: bool = False,
    covs: list[str] | None = None,
    cluster: bool = False,
    vce: str = "nn",
    c: float = 0.0,
    h: float | None = None,
    b: float | None = None,
    mask: pd.Series | None = None,
    bwselect: str | None = None,
) -> Fit:
    from rdrobust import rdrobust

    df = table if mask is None else table[mask]
    kw = _kwargs(params, df, y=y, fuzzy=fuzzy, covs=covs, cluster=cluster, vce=vce, c=c, bwselect=bwselect)
    if h is not None:
        kw["h"] = float(h)
    if b is not None:
        kw["b"] = float(b)
    try:
        est, notes = _run(rdrobust, **kw)
    except Exception as ex:
        return Fit(error=f"{type(ex).__name__}: {str(ex)[:300]}")
    return _convert(est, notes)


def _convert(est, notes: list[str]) -> Fit:
    try:
        f = Fit(
            value=float(est.coef.iloc[POINT_ROW, 0]),
            ci_low=float(est.ci.iloc[INTERVAL_ROW, 0]),
            ci_high=float(est.ci.iloc[INTERVAL_ROW, 1]),
            p=float(est.pv.iloc[INTERVAL_ROW, 0]),
            se=float(est.se.iloc[POINT_ROW, 0]),
            h_left=float(est.bws.loc["h", "left"]),
            h_right=float(est.bws.loc["h", "right"]),
            b_left=float(est.bws.loc["b", "left"]),
            b_right=float(est.bws.loc["b", "right"]),
            n_left=int(est.N[0]),
            n_right=int(est.N[1]),
            n_h_left=int(est.N_h[0]),
            n_h_right=int(est.N_h[1]),
            vce=str(est.vce),
            model=str(est.rdmodel),
            notes=notes,
        )
    except Exception as ex:
        return Fit(error=f"could not read the result: {type(ex).__name__}: {str(ex)[:200]}", notes=notes)
    if getattr(est, "tau_T", None) is not None:
        try:
            z = float(est.z_T.iloc[INTERVAL_ROW, 0])
            f.first_stage = dict(
                value=float(est.tau_T.iloc[POINT_ROW, 0]),
                ci_low=float(est.ci_T.iloc[INTERVAL_ROW, 0]),
                ci_high=float(est.ci_T.iloc[INTERVAL_ROW, 1]),
                z=z,
                se=float(est.se_T.iloc[POINT_ROW, 0]),
            )
        except Exception:
            f.first_stage = None
    if not all(np.isfinite(v) for v in (f.value, f.ci_low, f.ci_high, f.se)):
        f.error = "the fit returned a non-finite estimate, interval, or standard error"
    return f


def bandwidths(params: dict, table: pd.DataFrame, *, fuzzy: bool = False, covs: list[str] | None = None, cluster: bool = False, vce: str = "nn") -> dict:
    """Every selector the library offers, with the same arguments the primary fit uses."""
    from rdrobust import rdbwselect

    kw = _kwargs(params, table, y="y", fuzzy=fuzzy, covs=covs, cluster=cluster, vce=vce, c=0.0, bwselect=None)
    kw.pop("level", None)  # rdbwselect has no level argument
    kw["all"] = True
    try:
        bw, notes = _run(rdbwselect, **kw)
        t = bw.bws
        return dict(
            h_mse=float(t.loc["mserd", "h (left)"]),
            b_mse=float(t.loc["mserd", "b (left)"]),
            h_cer=float(t.loc["cerrd", "h (left)"]),
            b_cer=float(t.loc["cerrd", "b (left)"]),
            n_h_left=int(bw.N_h[0]),
            n_h_right=int(bw.N_h[1]),
            notes=notes,
            table={str(i): [float(v) for v in t.loc[i].to_numpy()] for i in t.index},
        )
    except Exception as ex:
        return dict(error=f"{type(ex).__name__}: {str(ex)[:300]}")


def density(x: np.ndarray, c: float = 0.0, floor: int = 23) -> dict:
    """The manipulation test on the scores as recorded (recentred, not flipped: with mass points the library's
    statistic depends on orientation, so the recorded one is used). Reads only the fields the library fills."""
    from rddensity import rddensity

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n_left, n_right = int((x < c).sum()), int((x >= c).sum())
    base = dict(n_left=n_left, n_right=n_right, floor=floor)
    if n_left < floor or n_right < floor:
        return dict(computable=False, reason=f"fewer than {floor} rows on a side ({n_left}/{n_right}); the library's own floor", **base)
    try:
        o, notes = _run(rddensity, X=x, c=c)
        t, p = float(o.test["t_jk"]), float(o.test["p_jk"])
        if p == 0.0 and np.isfinite(t):
            p = float(2 * norm.sf(abs(t)))
        h_left, h_right = float(o.h["left"]), float(o.h["right"])
        range_left, range_right = c - float(o.X_min["left"]), float(o.X_max["right"]) - c
        swallowed = h_left >= range_left or h_right >= range_right
        out = dict(
            t=t,
            p=p,
            h_left=h_left,
            h_right=h_right,
            range_left=range_left,
            range_right=range_right,
            eff_left=int(o.n["eff_left"]),
            eff_right=int(o.n["eff_right"]),
            hat_left=float(o.hat["left"]),
            hat_right=float(o.hat["right"]),
            mass_points=bool(o.massPoints_flag),
            notes=notes,
            **base,
        )
        if not np.isfinite(t):
            return dict(computable=False, reason="the statistic is not finite (too few distinct scores for the local fit)", **out)
        if swallowed:
            return dict(computable=False, reason="the bandwidth reaches a whole side of the data, so the test is a global comparison, not a local one", **out)
        return dict(computable=True, reason="", **out)
    except Exception as ex:
        return dict(computable=False, reason=f"{type(ex).__name__}: {str(ex)[:200]}", **base)


def bins(y: pd.Series, x: pd.Series, c: float = 0.0) -> pd.DataFrame | None:
    """rdplot's binned means, without drawing."""
    from rdrobust import rdplot

    try:
        rp, _ = _run(rdplot, y=y, x=x, c=c, hide=True)
        return rp.vars_bins
    except Exception:
        return None


def to_estimate(f: Fit, contrast_key: str, method: str, target_units: str, *, secondary: bool = False) -> Estimate:
    if f.error:
        return Estimate(contrast=contrast_key, method=method, target_units=target_units, secondary=secondary, error=f.error)
    return Estimate(
        contrast=contrast_key,
        method=method,
        value=f.value,
        ci_low=f.ci_low,
        ci_high=f.ci_high,
        n_treated=f.n_h_right,
        n_control=f.n_h_left,
        target_units=target_units,
        secondary=secondary,
    )
