"""Design checks on the canonical table: facts, flagged against checks.yaml. No model.

Every check is a CheckResult with an address the assess and interpret judgements can cite. The extra dict
carries facts the freeze needs (bandwidths, the first stage) so nothing is computed twice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency

from causal_agent.common.contracts import CheckResult
from causal_agent.families.discontinuity.lane import adapter
from causal_agent.families.discontinuity.lane.contracts import BalanceFacts, BalanceItem, BinomialWindow, Covariates, DensityFacts, ShapeFacts
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


def window_table(
    params: dict,
    canon: pd.DataFrame,
    shape: ShapeFacts,
    cfg: dict[str, Any],
    *,
    fuzzy: bool,
    covs: list[str] | None,
    cluster: bool,
    vce: str,
    sharpbw: bool = False,
) -> dict[str, Any]:
    """What the window rung reads: every selector the catalogue offers with the widths the library picks for this spec on each
    side and the rows each leaves inside, or the one support-points window when the score has few distinct values. Under the
    support-points rule there is nothing to judge, and the table says so with `rule`."""
    sup = cfg["support"]
    win = cfg["window"]
    x = canon["x"].to_numpy(dtype=float)

    def inside(h_left: float, h_right: float) -> tuple[int, int]:
        return int(((x < 0) & (x > -h_left)).sum()), int(((x >= 0) & (x < h_right)).sum())

    if shape.distinct_scores < int(sup["distinct_min"]["soft"]):
        plan = bandwidth_plan(params, canon, shape, cfg, fuzzy=fuzzy, cluster=cluster, vce=vce)
        if "error" in plan:
            return dict(error=plan["error"], rows={}, rule="support_points", default="support_points")
        h = float(plan["h"])
        n_l, n_r = inside(h, h)
        row = dict(rule="support_points", h_left=h, h_right=h, b_left=2 * h, b_right=2 * h, n_left=n_l, n_right=n_r)
        return dict(rule="support_points", default="support_points", rows={"support_points": row}, notes=plan.get("notes") or [])
    bw = adapter.bandwidths(params, canon, fuzzy=fuzzy, covs=covs, cluster=cluster, vce=vce, sharpbw=sharpbw)
    if "error" in bw:
        return dict(error=bw["error"], rows={}, rule="mse", default=win["default"])
    rows: dict[str, dict[str, Any]] = {}
    for name, spec in (win.get("selectors") or {}).items():
        t = bw["table"].get(name)
        if t is None:
            continue
        n_l, n_r = inside(t["h_left"], t["h_right"])
        rows[name] = dict(rule=spec["rule"], n_left=n_l, n_right=n_r, **t)
    return dict(rule="mse", default=win["default"], rows=rows, notes=bw.get("notes") or [])


def effective_rows_check(n_left: int, n_right: int, selector: str, h_left: float, h_right: float, contrast_key: str, cfg: dict[str, Any]) -> CheckResult:
    """The rows the primary fit uses on each side inside the window the rung chose."""
    e = int(cfg["effective_rows"]["min"]["soft"])
    smallest = min(n_left, n_right)
    width = f"h = {h_left:.4g}" if abs(h_left - h_right) < 1e-12 else f"h = {h_left:.4g}/{h_right:.4g}"
    return CheckResult(
        contrast=contrast_key,
        name="effective_rows",
        level="soft" if smallest < e else "pass",
        value=float(smallest),
        threshold=float(e),
        detail=f"{n_left} control-side and {n_right} treated-side rows inside the window {selector} ({width})",
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


def balance_evidence(
    canon: pd.DataFrame,
    others: pd.DataFrame,
    candidates: list[str],
    shape: ShapeFacts,
    cfg: dict[str, Any],
    *,
    cluster: bool,
    vce: str,
    window: float | None,
) -> BalanceFacts:
    """The balance rung, by code before the covariates are placed: every candidate's standing at the line. A number gets the
    sharp local linear jump at the coverage-error width (the continuity check's own fit, run here once); a category gets the
    difference in the share of its commonest level between the sides within `window` of the line, tested as a 2 by 2 table."""
    thr = float(cfg["covariate_continuity"]["p_value"]["soft"])
    plan = bandwidth_plan(SHARP, canon, shape, cfg, fuzzy=False, cluster=cluster, vce=vce)
    pinned = fixed(plan) if "error" not in plan else {}
    items: list[BalanceItem] = []
    for k in candidates:
        col = covcol(k)
        if col in canon.columns:
            f = adapter.fit(
                SHARP, canon, y=col, cluster=cluster, vce=vce, bwselect=None if pinned else cfg["covariate_continuity"].get("bwselect", "cerrd"), **pinned
            )
            if f.error:
                items.append(BalanceItem(column=k, how="untested", error=f.error))
            else:
                items.append(
                    BalanceItem(
                        column=k, how="jump", jump=f.value, ci_low=f.ci_low, ci_high=f.ci_high, p=f.p, n_left=f.n_h_left, n_right=f.n_h_right, width=f.h
                    )
                )
        elif k in others.columns:
            items.append(_share_item(k, others[k], canon["x"], window))
    return BalanceFacts(items=items, threshold=thr)


def _share_item(k: str, s: pd.Series, x: pd.Series, window: float | None) -> BalanceItem:
    w = float(window) if window is not None and np.isfinite(window) and window > 0 else float(np.nanmax(np.abs(x.to_numpy(dtype=float))) or 1.0)
    near = x.abs() < w
    v = s[near].astype(str).where(s[near].notna())
    side = (x[near] >= 0).to_numpy()
    v = v.to_numpy()
    keep = pd.notna(v)
    v, side = v[keep], side[keep]
    if len(v) == 0 or side.all() or not side.any():
        return BalanceItem(column=k, how="untested", error="no rows on one side within the window")
    top = pd.Series(v).value_counts().index[0]
    a, b = v[~side] == top, v[side] == top
    n_l, n_r = int((~side).sum()), int(side.sum())
    table = np.array([[a.sum(), n_l - a.sum()], [b.sum(), n_r - b.sum()]], dtype=float)
    try:
        p = float(chi2_contingency(table)[1]) if table.min() >= 0 and (table.sum(axis=0) > 0).all() else None
    except ValueError:
        p = None
    return BalanceItem(column=k, how="share", jump=float(b.mean() - a.mean()), p=p, n_left=n_l, n_right=n_r, width=w, level=str(top))


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
    balance: BalanceFacts,
) -> tuple[list[CheckResult], dict[str, Any]]:
    out: list[CheckResult] = []
    extra: dict[str, Any] = {}
    c = contrast_key

    # clusters: few of them make the clustered variance unreliable; inference.yaml switches its correction below the line
    if shape.clusters is not None:
        few = int(cfg["clusters"]["few"]["soft"])
        out.append(
            CheckResult(
                contrast=c,
                name="few_clusters",
                level="soft" if shape.clusters < few else "pass",
                value=float(shape.clusters),
                threshold=float(few),
                detail=f"{shape.clusters} clusters of {shape.cluster_column!r}"
                + ("; below the line where the cluster-robust variance is trusted" if shape.clusters < few else ""),
            )
        )

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

    # the sharp local linear plan, for pinning the first stage under the support-points rule; the effective rows are the window rung's
    plan = bandwidth_plan(SHARP, canon, shape, cfg, fuzzy=False, cluster=cluster, vce=vce)
    extra["bandwidth_plan"] = plan

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
        if shape.kind == "fuzzy" and (shape.takeup_left == 0.0 or shape.takeup_right == 1.0):
            out.append(
                CheckResult(
                    contrast=c,
                    name="one_sided_takeup",
                    level="pass",
                    value=shape.takeup_left if shape.takeup_left == 0.0 else shape.takeup_right,
                    detail=f"take-up {shape.takeup_left:.2f} on the control side, {shape.takeup_right:.2f} on the treated side: it varies on one side only, so the width is selected as for a sharp design",
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

    # covariate continuity: the balance rung's jumps, read here, not fitted again
    if covs.balance_tested:
        out.append(_continuity(balance, covs.balance_tested, c, cfg, extra))
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


def _continuity(balance: BalanceFacts, columns: list[str], c: str, cfg: dict, extra: dict) -> CheckResult:
    thr = cfg["covariate_continuity"]["p_value"]["soft"]
    rows: list[str] = []
    failed: list[str] = []
    per: dict[str, dict] = {}
    for col in columns:
        i = balance.item(col)
        if i is None or i.how != "jump" or i.p is None:
            rows.append(f"{col}: could not be tested ({i.error if i is not None and i.error else 'no jump on the balance rung'})")
            continue
        per[col] = dict(jump=i.jump, ci_low=i.ci_low, ci_high=i.ci_high, p=i.p, n_h_left=i.n_left, n_h_right=i.n_right, h=i.width)
        rows.append(f"{col}: jump {i.jump:.3g} at the cutoff (robust p = {i.p:.3g}, {i.n_left}/{i.n_right} rows) [ladder:balance.{col}]")
        if i.p < thr:
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
