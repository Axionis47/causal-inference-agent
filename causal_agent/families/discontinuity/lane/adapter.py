"""The only file that imports rdrobust, rddensity and rdlocrand. Reads a spec and the canonical table, returns facts.

One fit function for every local polynomial spec. Inference is a vce and a cluster passed through. Falsifications refit
the same function on a subset or with a fixed bandwidth. Local randomisation (a discrete score) has its own fit: the
difference in means in a window either side of the line, with a randomisation p-value and an interval found by inverting
that test over a grid in code, because the library's own inversion does not run in this port. Failures come back as
facts on the record, never as exceptions. Every call runs with numpy errors silenced (the Mac BLAS build warns on every
matmul), warnings captured as notes, and stdout captured for the library's printed diagnostics.
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
    h: float | list[float] | tuple[float, float] | None = None,
    b: float | list[float] | tuple[float, float] | None = None,
    mask: pd.Series | None = None,
    bwselect: str | None = None,
) -> Fit:
    """`h` and `b` are one width for both sides or a (control side, treated side) pair, in the score's units."""
    from rdrobust import rdrobust

    df = table if mask is None else table[mask]
    kw = _kwargs(params, df, y=y, fuzzy=fuzzy, covs=covs, cluster=cluster, vce=vce, c=c, bwselect=bwselect)
    if h is not None:
        kw["h"] = _sides(h)
    if b is not None:
        kw["b"] = _sides(b)
    try:
        est, notes = _run(rdrobust, **kw)
    except Exception as ex:
        return Fit(error=f"{type(ex).__name__}: {str(ex)[:300]}")
    return _convert(est, notes)


def _sides(v: float | list[float] | tuple[float, float]) -> float | list[float]:
    if isinstance(v, (list, tuple)):
        left, right = float(v[0]), float(v[1])
        return left if abs(left - right) < 1e-12 else [left, right]
    return float(v)


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
            table={
                str(i): dict(
                    h_left=float(t.loc[i, "h (left)"]),
                    h_right=float(t.loc[i, "h (right)"]),
                    b_left=float(t.loc[i, "b (left)"]),
                    b_right=float(t.loc[i, "b (right)"]),
                )
                for i in t.index
            },
        )
    except Exception as ex:
        return dict(error=f"{type(ex).__name__}: {str(ex)[:300]}")


def density(x: np.ndarray, c: float = 0.0, floor: int = 23, params: dict | None = None) -> dict:
    """The manipulation test on the scores as recorded (recentred, not flipped: with mass points the library's
    statistic depends on orientation, so the recorded one is used), with the settings checks.yaml declares (the
    polynomial order, the kernel, the variance, the bandwidth selector, the model). Reads only the fields the library fills."""
    from rddensity import rddensity

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n_left, n_right = int((x < c).sum()), int((x >= c).sum())
    base = dict(n_left=n_left, n_right=n_right, floor=floor)
    if n_left < floor or n_right < floor:
        return dict(computable=False, reason=f"fewer than {floor} rows on a side ({n_left}/{n_right}); the library's own floor", **base)
    kw: dict[str, Any] = {k: v for k, v in (params or {}).items() if k in ("p", "kernel", "vce", "bwselect", "fitselect")}
    try:
        o, notes = _run(rddensity, X=x, c=c, **kw)
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


# ------------------------------------------------------------------ local randomisation (rdlocrand)

GRID_POINTS = 41
GRID_HALF_WIDTHS = 4.0  # the inversion grid reaches this many standard errors either side of the point


def local_random_windows(canon: pd.DataFrame, covs: list[str], *, reps: int, seed: int) -> dict:
    """Nested windows around the line ranked by covariate balance (rdwinselect): the recommended window, the largest whose
    balance holds in every smaller one, and every candidate with its smallest balance p, the binomial test on the score and the
    rows a side. With no covariates the library recommends none, and the caller falls back to the support-points window."""
    from rdlocrand import rdwinselect

    cols = [c for c in covs if c in canon.columns]
    kw: dict[str, Any] = dict(R=canon["x"].to_numpy(dtype=float), reps=int(reps), seed=int(seed), quietly=True, wmasspoints=True, dropmissing=True)
    if cols:
        kw["X"] = canon[cols].to_numpy(dtype=float)
    try:
        o, notes = _run(rdwinselect, **kw)
        rows = []
        for _, r in o["results"].iterrows():
            p_min = float(r["p-value"])
            rows.append(
                dict(
                    w_left=float(r["w_left"]),
                    w_right=float(r["w_right"]),
                    p_min=None if np.isnan(p_min) else p_min,
                    binomial_p=float(r["Bi.test"]),
                    n_left=int(r["Obs<c"]),
                    n_right=int(r["Obs>=c"]),
                )
            )
        wl, wr = o.get("w_left"), o.get("w_right")
        ok = wl is not None and wr is not None and np.isfinite(wl) and np.isfinite(wr)
        return dict(w_left=float(wl) if ok else None, w_right=float(wr) if ok else None, windows=rows, covariates=cols, notes=notes)
    except Exception as ex:
        return dict(error=f"{type(ex).__name__}: {str(ex)[:300]}", windows=[], covariates=cols)


def _randinf(canon: pd.DataFrame, wl: float, wr: float, *, fuzzy: bool, reps: int, seed: int, nulltau: float = 0.0) -> dict:
    from rdlocrand import rdrandinf

    kw: dict[str, Any] = dict(
        Y=canon["y"].to_numpy(dtype=float), R=canon["x"].to_numpy(dtype=float), wl=float(wl), wr=float(wr), reps=int(reps), seed=int(seed), quietly=True
    )
    if nulltau:
        kw["nulltau"] = float(nulltau)
    if fuzzy:
        kw["fuzzy"] = [canon["t"].to_numpy(dtype=float), "ar"]  # the Anderson-Rubin statistic, the one that inverts under a null effect
    o, _ = _run(rdrandinf, **kw)
    return o


def fit_local_random(canon: pd.DataFrame, wl: float, wr: float, *, fuzzy: bool = False, reps: int = 1000, seed: int = 7, alpha: float = 0.05) -> Fit:
    """The local randomisation estimate in the window [wl, wr] of the recentred score: the difference in means (sharp) or the
    Wald ratio of the outcome's jump to take-up's jump (fuzzy), the randomisation p-value under no effect, and the interval of
    effects the randomisation test does not reject at `alpha`, found by inverting the test over a grid."""
    try:
        base = _randinf(canon, wl, wr, fuzzy=fuzzy, reps=reps, seed=seed)
    except Exception as ex:
        return Fit(error=f"{type(ex).__name__}: {str(ex)[:300]}")
    try:
        p = float(base["p.value"])
        s = np.asarray(base["sumstats"], dtype=float)
        n_left, n_right = int(s[0][0]), int(s[0][1])
        n_w_left, n_w_right = int(s[1][0]), int(s[1][1])
        mean_l, mean_r, sd_l, sd_r = float(s[2][0]), float(s[2][1]), float(s[3][0]), float(s[3][1])
    except Exception as ex:
        return Fit(error=f"could not read the result: {type(ex).__name__}: {str(ex)[:200]}")
    if n_w_left < 2 or n_w_right < 2:
        return Fit(error=f"the window holds {n_w_left} rows on the control side and {n_w_right} on the treated side; too few to compare")
    inside = (canon["x"] >= wl) & (canon["x"] <= wr)
    value = mean_r - mean_l
    se = float(np.sqrt(sd_l**2 / n_w_left + sd_r**2 / n_w_right))
    if fuzzy:
        t = canon.loc[inside, "t"]
        first = float(t[canon.loc[inside, "x"] >= 0].mean() - t[canon.loc[inside, "x"] < 0].mean())
        if not np.isfinite(first) or abs(first) < 1e-9:
            return Fit(error="take-up does not jump inside the window, so the complier effect is not defined")
        value, se = value / first, se / abs(first)
    half = max(GRID_HALF_WIDTHS * se, 1e-9)
    lo = hi = None
    for _ in range(3):  # widen the grid when the accepted set touches its ends
        grid = np.linspace(value - half, value + half, GRID_POINTS)
        try:
            accepted = [float(tau) for tau in grid if float(_randinf(canon, wl, wr, fuzzy=fuzzy, reps=reps, seed=seed, nulltau=float(tau))["p.value"]) >= alpha]
        except Exception as ex:
            return Fit(error=f"the interval could not be inverted: {type(ex).__name__}: {str(ex)[:200]}")
        if not accepted:
            break
        lo, hi = min(accepted), max(accepted)
        if lo > grid[0] and hi < grid[-1]:
            break
        half *= 2
    if lo is None or hi is None:
        return Fit(error="no effect on the grid is accepted by the randomisation test; the interval could not be found")
    return Fit(
        value=float(value),
        ci_low=float(lo),
        ci_high=float(hi),
        p=p,
        se=se,
        h_left=float(-wl),
        h_right=float(wr),
        n_left=n_left,
        n_right=n_right,
        n_h_left=n_w_left,
        n_h_right=n_w_right,
        vce="randomisation",
        model="local randomisation, " + ("Anderson-Rubin" if fuzzy else "difference in means"),
        notes=[f"{reps} permutations; the interval inverts the test at {alpha:g} over {GRID_POINTS} points"],
    )


def window_sensitivity(canon: pd.DataFrame, windows: list[tuple[float, float]], *, fuzzy: bool, reps: int, seed: int, alpha: float = 0.05) -> list[dict]:
    """The local randomisation estimate across windows: one row per window with the estimate, its interval, its p and the rows a side."""
    out = []
    for wl, wr in windows:
        f = fit_local_random(canon, wl, wr, fuzzy=fuzzy, reps=reps, seed=seed, alpha=alpha)
        out.append(
            dict(
                w_left=float(wl),
                w_right=float(wr),
                value=f.value,
                ci_low=f.ci_low,
                ci_high=f.ci_high,
                p=f.p,
                n_left=f.n_h_left,
                n_right=f.n_h_right,
                error=f.error,
            )
        )
    return out


def rosenbaum_bounds(canon: pd.DataFrame, wr: float, gammas: list[float], *, reps: int, seed: int) -> dict:
    """Rosenbaum bounds on the randomisation p-value in the symmetric window of half-width `wr`: for each gamma, the odds by
    which the assignment inside the window may depart from a coin toss, the lowest and highest p-value it allows."""
    from rdlocrand import rdrbounds

    try:
        o, notes = _run(
            rdrbounds,
            Y=canon["y"].to_numpy(dtype=float),
            R=canon["x"].to_numpy(dtype=float),
            wlist=np.array([float(wr)]),
            gamma=np.array([float(g) for g in gammas]),
            reps=int(reps),
            seed=int(seed),
        )
        return dict(
            gamma=[float(g) for g in np.ravel(o["gamma"])],
            p=float(np.ravel(o["p.values"])[0]),
            lower=[float(v) for v in np.ravel(o["lower.bound"])],
            upper=[float(v) for v in np.ravel(o["upper.bound"])],
            notes=notes,
        )
    except Exception as ex:
        return dict(error=f"{type(ex).__name__}: {str(ex)[:300]}")


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
