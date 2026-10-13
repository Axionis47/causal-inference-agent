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
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from scipy.stats import norm

from causal_agent.common.contracts import ConfidenceSet, Estimate

POINT_ROW, INTERVAL_ROW = 2, 2  # paired robust bias-corrected estimate, SE, interval and p-value


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
    confidence_set: ConfidenceSet | None = None
    inference_row: str = "robust"

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


_QUIET = threading.Lock()  # redirecting stdout and recording warnings swap process-wide state; two placebo workers must not interleave


def _run(fn, **kw):
    buf = io.StringIO()
    with _QUIET, np.errstate(all="ignore"), warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(buf):
        warnings.simplefilter("always")
        res = fn(**kw)
    notes = [str(w.message) for w in caught if not issubclass(w.category, RuntimeWarning)]
    notes += [line.strip() for line in buf.getvalue().splitlines() if line.strip()]
    return res, notes


def _kwargs(
    params: dict,
    df: pd.DataFrame,
    *,
    y: str,
    fuzzy: bool,
    covs: list[str] | None,
    cluster: bool,
    vce: str,
    c: float,
    bwselect: str | None,
    sharpbw: bool = False,
) -> dict:
    kw: dict[str, Any] = dict(
        y=df[y],
        x=df["x"],
        c=c,
        p=int(params.get("p", 1)),
        q=int(params.get("q", int(params.get("p", 1)) + 1)),
        kernel=params.get("kernel", "tri"),
        bwselect=bwselect or params.get("bwselect", "mserd"),
        vce=vce,
        masspoints=params.get("masspoints", "adjust"),
        level=int(params.get("level", 95)),
    )
    if fuzzy:
        kw["fuzzy"] = df["t"]
        if sharpbw:
            kw["sharpbw"] = True  # take-up varies on one side only: the library selects the width as for a sharp design
    if covs:
        kw["covs"] = df[list(covs)]
        kw["covs_drop"] = False
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
    sharpbw: bool = False,
) -> Fit:
    """`h` and `b` are one width for both sides or a (control side, treated side) pair, in the score's units; `sharpbw` asks the
    library to select the width as for a sharp design when take-up varies on one side only."""
    from rdrobust import rdrobust

    df = table if mask is None else table[mask]
    kw = _kwargs(params, df, y=y, fuzzy=fuzzy, covs=covs, cluster=cluster, vce=vce, c=c, bwselect=bwselect, sharpbw=sharpbw)
    if h is not None:
        kw["h"] = _sides(h)
    if b is not None:
        kw["b"] = _sides(b)
    try:
        est, notes = _run(rdrobust, **kw)
    except Exception as ex:
        return Fit(error=f"{type(ex).__name__}: {str(ex)[:300]}")
    return _convert(est, notes, row={"conventional": 0, "bias_corrected": 1, "robust": 2}.get(params.get("inference_row", "robust"), 2))


def _sides(v: float | list[float] | tuple[float, float]) -> float | list[float]:
    if isinstance(v, (list, tuple)):
        left, right = float(v[0]), float(v[1])
        return left if abs(left - right) < 1e-12 else [left, right]
    return float(v)


def _convert(est, notes: list[str], row: int = 2) -> Fit:
    try:
        f = Fit(
            value=float(est.coef.iloc[row, 0]),
            ci_low=float(est.ci.iloc[row, 0]),
            ci_high=float(est.ci.iloc[row, 1]),
            p=float(est.pv.iloc[row, 0]),
            se=float(est.se.iloc[row, 0]),
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
            inference_row=("conventional", "bias_corrected", "robust")[row],
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


def bandwidths(
    params: dict, table: pd.DataFrame, *, fuzzy: bool = False, covs: list[str] | None = None, cluster: bool = False, vce: str = "nn", sharpbw: bool = False
) -> dict:
    """Every selector the library offers, with the same arguments the primary fit uses."""
    from rdrobust import rdbwselect

    kw = _kwargs(params, table, y="y", fuzzy=fuzzy, covs=covs, cluster=cluster, vce=vce, c=0.0, bwselect=None, sharpbw=sharpbw)
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


def accepted_regions(grid: np.ndarray, accepted: list[bool]) -> list[list[float]]:
    """Every connected accepted grid component. Never fill a rejected gap."""
    regions = []
    start = None
    for i, ok in enumerate(accepted):
        if ok and start is None:
            start = i
        if start is not None and (not ok or i == len(grid) - 1):
            end = i if ok else i - 1
            regions.append([float(grid[start]), float(grid[end])])
            start = None
    return regions


def fit_local_random(
    canon: pd.DataFrame, wl: float, wr: float, *, fuzzy: bool = False, reps: int = 1000, seed: int = 7, alpha: float = 0.05, mask: pd.Series | None = None
) -> Fit:
    """Fixed-window test inversion. Weak receipt leaves a confidence set, not an invented Wald interval."""
    if mask is not None:
        canon = canon[mask]
    inside = (canon.x >= wl) & (canon.x <= wr)
    sample = canon[inside]
    left, right = sample[sample.x < 0], sample[sample.x >= 0]
    nl, nr = len(left), len(right)
    if nl < 2 or nr < 2:
        return Fit(error="the frozen window has fewer than two observations on a side")
    try:
        base = _randinf(canon, wl, wr, fuzzy=fuzzy, reps=reps, seed=seed)
        p = float(base["p.value"])
    except Exception as ex:
        return Fit(error=f"randomisation test failed: {type(ex).__name__}: {str(ex)[:250]}")
    reduced = float(right.y.mean() - left.y.mean())
    se = float(np.sqrt(left.y.var() / nl + right.y.var() / nr))
    first = float(right.t.mean() - left.t.mean()) if fuzzy else 1.0
    value = reduced / first if abs(first) > 1e-9 else None
    # Grid units are outcome units for sharp and outcome per receipt for fuzzy. Its bounds are a computation limit.
    centre = value if value is not None and np.isfinite(value) else 0.0
    spread = se / abs(first) if abs(first) > 1e-9 else max(float(sample.y.std()), 1.0)
    half = max(GRID_HALF_WIDTHS * spread, abs(centre) * 0.5, 1e-6)
    accepted = []
    grid = np.array([])
    for _ in range(3):
        grid = np.linspace(centre - half, centre + half, GRID_POINTS)
        try:
            accepted = [float(_randinf(canon, wl, wr, fuzzy=fuzzy, reps=reps, seed=seed, nulltau=float(tau))["p.value"]) >= alpha for tau in grid]
        except Exception as ex:
            cs = ConfidenceSet(status="computational_failure", searched_range=[float(grid[0]), float(grid[-1])], detail=str(ex)[:250])
            return Fit(value=value, p=p, n_h_left=nl, n_h_right=nr, confidence_set=cs, error="confidence-set inversion failed")
        if not accepted[0] and not accepted[-1]:
            break
        half *= 2
    regions = accepted_regions(grid, accepted)
    status = "empty" if not regions else "unresolved_tail" if accepted[0] or accepted[-1] else "disconnected" if len(regions) > 1 else "bounded"
    cs = ConfidenceSet(
        status=status,
        regions=regions,
        searched_range=[float(grid[0]), float(grid[-1])],
        resolution=float(grid[1] - grid[0]),
        detail=(
            "grid inversion of the additive-effect null; the searched boundary was accepted, leaving that outer tail unresolved"
            if status == "unresolved_tail"
            else "grid inversion of the additive-effect null; both searched boundaries were rejected; the reported regions describe the finite grid at the recorded resolution"
        ),
    )
    lo, hi = regions[0] if status == "bounded" else (None, None)
    return Fit(
        value=value,
        ci_low=lo,
        ci_high=hi,
        p=p,
        se=spread,
        h_left=-wl,
        h_right=wr,
        n_left=nl,
        n_right=nr,
        n_h_left=nl,
        n_h_right=nr,
        vce="randomisation",
        model="Anderson-Rubin in the frozen window" if fuzzy else "difference in means in the frozen window",
        confidence_set=cs,
        first_stage={"value": first} if fuzzy else None,
        notes=[f"{reps} permutations; {GRID_POINTS} grid points; {status}"],
        inference_row="randomisation",
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
                confidence_set=f.confidence_set,
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
        return Estimate(contrast=contrast_key, method=method, target_units=target_units, secondary=secondary, error=f.error, confidence_set=f.confidence_set)
    return Estimate(
        contrast=contrast_key,
        method=method,
        confidence_set=f.confidence_set,
        value=f.value,
        ci_low=f.ci_low,
        ci_high=f.ci_high,
        n_treated=f.n_h_right,
        n_control=f.n_h_left,
        target_units=target_units,
        secondary=secondary,
        p_value=f.p,
        p_value_source="randomisation" if f.vce == "randomisation" else f"{f.inference_row} ({f.vce})" if f.vce else None,
    )
