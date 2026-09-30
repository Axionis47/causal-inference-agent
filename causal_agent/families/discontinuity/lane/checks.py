"""Design checks on the canonical table: facts, flagged against checks.yaml. No model.

Every check is a CheckResult with an address the assess and interpret judgements can cite. The extra dict
carries facts the freeze needs (bandwidths, the first stage) so nothing is computed twice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from causal_agent.common.contracts import CheckResult
from causal_agent.families.discontinuity.lane import adapter
from causal_agent.families.discontinuity.lane.contracts import BinomialWindow, Covariates, DensityFacts, ShapeFacts
from causal_agent.families.discontinuity.lane.shape import covcol

SHARP = {"p": 1, "kernel": "tri", "bwselect": "mserd", "masspoints": "adjust", "level": 95}


def bandwidth_plan(params: dict, canon: pd.DataFrame, shape: ShapeFacts, cfg: dict[str, Any], *, fuzzy: bool, cluster: bool, vce: str) -> dict[str, Any]:
    """The bandwidths a spec uses on this table: the library's MSE choice, or, when the score has few distinct
    values, the distance that keeps a declared number of support points on each side (a declared rule, not a fit)."""
    sup = cfg["support"]
    if shape.distinct_scores < int(sup["distinct_min"]["soft"]):
        k = int(sup.get("bandwidth_support_points", 3))
        x = canon["x"]
        sides = []
        for vals in (np.sort(np.abs(x[x < 0].unique())), np.sort(x[x >= 0].unique())):
            vals = vals[vals > 0] if len(vals) and vals[0] == 0 and len(vals) > 1 else vals
            if len(vals) == 0:
                sides.append(float("nan"))
            elif len(vals) <= k:
                sides.append(float(vals[-1]) * 1.5)
            else:
                sides.append(float((vals[k - 1] + vals[k]) / 2))
        h = float(np.nanmax(sides))
        if not np.isfinite(h) or h <= 0:
            return dict(error="the score has no support on one side")
        n_l, n_r = int(((x < 0) & (x > -h)).sum()), int(((x >= 0) & (x < h)).sum())
        return dict(rule="support_points", h=h, b=2 * h, h_cer=None, n_h_left=n_l, n_h_right=n_r, notes=[f"{k} support points a side"])
    bw = adapter.bandwidths(params, canon, fuzzy=fuzzy, cluster=cluster, vce=vce)
    if "error" in bw:
        return bw
    return dict(
        rule="mse", h=bw["h_mse"], b=bw["b_mse"], h_cer=bw["h_cer"], n_h_left=bw["n_h_left"], n_h_right=bw["n_h_right"], notes=bw["notes"], table=bw["table"]
    )


def fixed(plan: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments that pin the bandwidths when the plan is not the library's own choice."""
    return {"h": plan["h"], "b": plan["b"]} if plan.get("rule") == "support_points" else {}


def density_evidence(x_all: pd.Series, shape: ShapeFacts, cfg: dict[str, Any], *, sampled_by_side: bool) -> tuple[DensityFacts, dict[str, Any]]:
    """The density rung, by code before the line is judged: the library's test with the declared settings, the binomial split
    of the rows in nested windows (shares of the test's smaller bandwidth), the histogram either side, the mass points. Returns
    the record for the ladder and the raw test the check and the figure read, so the library runs once."""
    dc = cfg["density"]
    x = pd.to_numeric(x_all, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    raw = adapter.density(x, floor=int(dc["library_floor_rows_per_side"]), params=dc.get("params") or {})
    bins = int(dc.get("histogram_bins", 10))
    hist = _histogram(x, bins)
    base = dict(
        histogram=hist,
        mass_share_left=shape.duplicate_share_left,
        mass_share_right=shape.duplicate_share_right,
        sampled_by_side=sampled_by_side,
    )
    if sampled_by_side:
        return DensityFacts(status="uninformative", reason="the rows were drawn by side of the line", **base), raw
    if not raw["computable"]:
        return DensityFacts(status="not_computable", reason=raw["reason"], **base), raw
    reach = min(float(raw["h_left"]), float(raw["h_right"]))
    windows = _binomial_windows(x, reach, [float(v) for v in dc["binomial"]["window_shares"]])
    flagged = float(raw["p"]) < float(dc["p_value"]["soft"]) or (bool(windows) and windows[0].p < float(dc["binomial"]["p_value"]["soft"]))
    facts = DensityFacts(
        status="tested",
        p=float(raw["p"]),
        t=float(raw["t"]),
        hat_left=float(raw["hat_left"]),
        hat_right=float(raw["hat_right"]),
        h_left=float(raw["h_left"]),
        h_right=float(raw["h_right"]),
        n_eff_left=int(raw["eff_left"]),
        n_eff_right=int(raw["eff_right"]),
        windows=windows,
        flagged=flagged,
        **base,
    )
    return facts, raw


def _binomial_windows(x: np.ndarray, reach: float, shares: list[float]) -> list[BinomialWindow]:
    out: list[BinomialWindow] = []
    if not np.isfinite(reach) or reach <= 0:
        return out
    for s in shares:
        w = reach * s
        left, right = int(((x < 0) & (x >= -w)).sum()), int(((x >= 0) & (x <= w)).sum())
        if left + right == 0:
            continue
        p = float(binomtest(right, left + right, 0.5).pvalue)
        out.append(BinomialWindow(width=float(w), n_left=left, n_right=right, p=p))
    return out


def _histogram(x: np.ndarray, bins: int) -> list[tuple[float, float, int]]:
    if len(x) == 0:
        return []
    k = max(2, bins + bins % 2)
    reach = float(max(abs(x.min()), abs(x.max()))) or 1.0
    edges = np.linspace(-reach, reach, k + 1)
    counts, _ = np.histogram(x, bins=edges)
    return [(float(edges[i]), float(edges[i + 1]), int(counts[i])) for i in range(k)]


def run_checks(
    canon: pd.DataFrame,
    x_all: pd.Series,
    shape: ShapeFacts,
    covs: Covariates,
    contrast_key: str,
    cfg: dict[str, Any],
    *,
    cluster: bool,
    vce: str,
    sampled_by_side: bool,
    density: dict[str, Any],
) -> tuple[list[CheckResult], dict[str, Any]]:
    out: list[CheckResult] = []
    extra: dict[str, Any] = {}
    c = contrast_key

    # sides
    u = cfg["sides"]["min_rows"]
    smallest = min(shape.n_left, shape.n_right)
    out.append(
        CheckResult(
            contrast=c,
            name="sides",
            level="hard" if smallest < u["hard"] else "soft" if smallest < u["soft"] else "pass",
            value=float(smallest),
            threshold=float(u["soft"]),
            detail=f"{shape.n_left} rows on the control side, {shape.n_right} on the treated side",
        )
    )

    # effective rows at the bandwidth of the sharp local linear fit
    plan = bandwidth_plan(SHARP, canon, shape, cfg, fuzzy=False, cluster=cluster, vce=vce)
    extra["bandwidth_plan"] = plan
    if "error" in plan:
        out.append(CheckResult(contrast=c, name="effective_rows", level="soft", detail=f"bandwidth selection failed: {plan['error']}"))
    else:
        e = cfg["effective_rows"]["min"]["soft"]
        smallest_eff = min(plan["n_h_left"], plan["n_h_right"])
        rule = "the MSE bandwidth" if plan["rule"] == "mse" else "the support-points bandwidth (few distinct scores)"
        out.append(
            CheckResult(
                contrast=c,
                name="effective_rows",
                level="soft" if smallest_eff < e else "pass",
                value=float(smallest_eff),
                threshold=float(e),
                detail=f"{plan['n_h_left']} control-side and {plan['n_h_right']} treated-side rows inside {rule} h = {plan['h']:.4g}",
            )
        )

    # density: the rung's test, read here, not run again
    out.append(_density(density, c, cfg, sampled_by_side, extra))

    # mass points and support
    m = cfg["mass_points"]["duplicate_share"]["soft"]
    dup = max(shape.duplicate_share_left, shape.duplicate_share_right)
    out.append(
        CheckResult(
            contrast=c,
            name="mass_points",
            level="soft" if dup >= m else "pass",
            value=round(dup, 3),
            threshold=float(m),
            detail=f"duplicated scores: {shape.duplicate_share_left:.0%} on the control side, {shape.duplicate_share_right:.0%} on the treated side"
            + ("; the library adjusts its bandwidth floor for mass points" if dup >= m else ""),
        )
    )
    s = cfg["support"]["distinct_min"]["soft"]
    out.append(
        CheckResult(
            contrast=c,
            name="support",
            level="soft" if shape.distinct_scores < s else "pass",
            value=float(shape.distinct_scores),
            threshold=float(s),
            detail=f"{shape.distinct_scores} distinct scores" + ("; too few for bandwidth selection to mean much" if shape.distinct_scores < s else ""),
        )
    )

    # compliance and the first stage
    if "t" in canon.columns:
        out.append(
            CheckResult(
                contrast=c,
                name="compliance",
                level="pass",
                value=shape.takeup_right,
                detail=f"take-up {shape.takeup_left:.2f} on the control side, {shape.takeup_right:.2f} on the treated side: {shape.kind}",
            )
        )
        if shape.kind == "sharp":
            extra["first_stage_status"] = "strong"
            extra["first_stage_F"] = float("inf")
            out.append(
                CheckResult(
                    contrast=c, name="first_stage", level="pass", value=1.0, detail="take-up jumps from 0 to 1 at the cutoff by the rule itself; no fit needed"
                )
            )
        else:
            out += _first_stage(canon, c, cfg, cluster, vce, extra, fixed(plan) if "error" not in plan else {})
    else:
        extra["first_stage_status"] = None

    # covariate continuity
    if covs.balance_tested:
        out.append(_continuity(canon, covs.balance_tested, c, cfg, cluster, vce, extra, fixed(plan) if "error" not in plan else {}))
    return out, extra


def _density(d: dict, c: str, cfg: dict, sampled_by_side: bool, extra: dict) -> CheckResult:
    extra["density"] = d
    pop = f"population: {d.get('n_left', 0) + d.get('n_right', 0)} rows with a score as recorded, {d.get('n_left', 0)} below and {d.get('n_right', 0)} at or above the cutoff"
    if sampled_by_side:  # no value: the number says nothing here, and no rule may read it as a jump
        return CheckResult(
            contrast=c,
            name="density",
            level="soft",
            detail=f"uninformative: the rows were sampled by side of the cutoff, so their density says nothing about manipulation; {pop}",
        )
    if not d["computable"]:
        return CheckResult(contrast=c, name="density", level="soft", detail=f"not computable: {d['reason']}; {pop}")
    thr = cfg["density"]["p_value"]["soft"]
    w = d.get("windows") or []
    bthr = float(cfg["density"]["binomial"]["p_value"]["soft"])
    bunch = bool(w) and float(w[0]["p"]) < bthr
    level = "soft" if d["p"] < thr or bunch else "pass"
    return CheckResult(
        contrast=c,
        name="density",
        level=level,
        value=round(d["p"], 4),
        threshold=float(thr),
        detail=f"density {d['hat_left']:.3g} just below the cutoff, {d['hat_right']:.3g} just above; test p = {d['p']:.3g} on {d['eff_left']}/{d['eff_right']} effective rows; {pop}"
        + (f"; in the smallest window ±{w[0]['width']:.3g} the rows split {w[0]['n_left']} | {w[0]['n_right']} (coin-toss p = {w[0]['p']:.2g})" if w else "")
        + ("; a jump in the number of units at the cutoff" if level == "soft" else "; no sign of bunching"),
    )


def _first_stage(canon: pd.DataFrame, c: str, cfg: dict, cluster: bool, vce: str, extra: dict, pinned: dict) -> list[CheckResult]:
    fs = adapter.fit(SHARP, canon, y="t", cluster=cluster, vce=vce, **pinned)
    extra["first_stage"] = fs
    out: list[CheckResult] = []
    if fs.error or fs.covers_zero() is None:
        extra["first_stage_status"] = "none"
        out.append(
            CheckResult(
                contrast=c,
                name="no_first_stage",
                level="hard" if cfg["first_stage"]["no_first_stage_is_hard"] else "soft",
                detail=f"the jump in take-up at the cutoff could not be estimated: {fs.error}",
            )
        )
        return out
    jump = f"take-up jumps by {fs.value:.3f} at the cutoff, robust interval {fs.ci_low:.3f} to {fs.ci_high:.3f}"
    if fs.covers_zero():
        extra["first_stage_status"] = "none"
        out.append(
            CheckResult(
                contrast=c,
                name="no_first_stage",
                level="hard" if cfg["first_stage"]["no_first_stage_is_hard"] else "soft",
                value=fs.value,
                detail=f"{jump}: the cutoff does not move take-up, so it identifies nothing",
            )
        )
        return out
    z = fs.value / fs.se if fs.se and fs.se > 0 else float("inf")
    F = z * z
    strong = F >= cfg["first_stage"]["f_min"]
    extra["first_stage_status"] = "strong" if strong else "weak"
    extra["first_stage_F"] = F
    out.append(
        CheckResult(
            contrast=c,
            name="first_stage_weak" if not strong else "first_stage",
            level="pass" if strong else "soft",
            value=round(F, 2) if np.isfinite(F) else None,
            threshold=float(cfg["first_stage"]["f_min"]),
            detail=f"{jump}; F = {F:.1f}"
            if np.isfinite(F)
            else f"{jump}; take-up is a step function of the score"
            + ("" if strong else "; below the strength the Extensions require, so only the effect of crossing the cutoff is reported"),
        )
    )
    return out


def _continuity(canon: pd.DataFrame, columns: list[str], c: str, cfg: dict, cluster: bool, vce: str, extra: dict, pinned: dict) -> CheckResult:
    thr = cfg["covariate_continuity"]["p_value"]["soft"]
    rows: list[str] = []
    failed: list[str] = []
    per: dict[str, dict] = {}
    for col in columns:
        f = adapter.fit(
            SHARP, canon, y=covcol(col), cluster=cluster, vce=vce, bwselect=None if pinned else cfg["covariate_continuity"].get("bwselect", "cerrd"), **pinned
        )
        if f.error:
            rows.append(f"{col}: could not be tested ({f.error})")
            continue
        per[col] = dict(jump=f.value, ci_low=f.ci_low, ci_high=f.ci_high, p=f.p, n_h_left=f.n_h_left, n_h_right=f.n_h_right, h=f.h)
        rows.append(f"{col}: jump {f.value:.3g} at the cutoff (robust p = {f.p:.3g}, {f.n_h_left}/{f.n_h_right} rows)")
        if f.p < thr:
            failed.append(col)
    extra["continuity"] = per
    level = "soft" if failed else "pass"
    return CheckResult(
        contrast=c,
        name="covariate_continuity",
        level=level,
        value=float(len(failed)),
        threshold=float(thr),
        detail="; ".join(rows)
        + (
            f"; {', '.join(failed)} differ at the cutoff, which the design says they should not"
            if failed
            else "; every predetermined covariate is continuous at the cutoff"
        ),
    )
