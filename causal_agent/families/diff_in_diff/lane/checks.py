"""Design checks on the canonical panel: facts, flagged against checks.yaml. No model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from causal_agent.common.contracts import CheckResult
from causal_agent.families.diff_in_diff.lane import adapter
from causal_agent.families.diff_in_diff.lane.contracts import Composition, PathPoint, ShapeFacts, TrendFacts
from causal_agent.families.diff_in_diff.lane.shape import period_label


def trend_facts(panel: pd.DataFrame, shape: ShapeFacts, cfg: dict[str, Any]) -> tuple[TrendFacts, dict[str, Any]]:
    """The trends rung: what the panel shows before the earliest change, computed once without controls. Returns the record
    for the ladder and the raw leads fit (every period's coefficient) the check and the figure read after the freeze."""
    tcfg = cfg.get("trends") or {}
    pre = panel[panel["post"] == 0]
    by = pre.groupby(["time", "treated"])["y"].agg(["mean", "size"]).unstack("treated")
    points: list[PathPoint] = []
    for t in sorted(pre["time"].unique())[-int(tcfg.get("max_paths_periods", 12)) :]:
        tm = by["mean"].get(1, pd.Series(dtype=float)).get(t) if "mean" in by else None
        cm = by["mean"].get(0, pd.Series(dtype=float)).get(t) if "mean" in by else None
        points.append(
            PathPoint(
                time=period_label(t),
                treated_mean=None if tm is None or pd.isna(tm) else float(tm),
                control_mean=None if cm is None or pd.isna(cm) else float(cm),
                n_treated=int(by["size"].get(1, pd.Series(dtype=float)).get(t, 0) or 0) if "size" in by else 0,
                n_control=int(by["size"].get(0, pd.Series(dtype=float)).get(t, 0) or 0) if "size" in by else 0,
            )
        )
    gaps = [(i, pt.treated_mean - pt.control_mean) for i, pt in enumerate(points) if pt.treated_mean is not None and pt.control_mean is not None]
    slope = float(np.polyfit([i for i, _ in gaps], [g for _, g in gaps], 1)[0]) if len(gaps) >= 2 else None
    res, dyn, how, err = (
        _leads(panel, [], cfg, shape.cohorts, shape.never_treated_exists)
        if shape.periods_pre > int(cfg["periods"]["parallel_untestable_when_pre"])
        else (None, {}, "", "")
    )
    thr = cfg["pre_trends"]["p_value"]
    if res is None:
        level, stat, pv, k = "untested", None, None, 0
        how = err or how or f"{shape.periods_pre} period before the change; the paths cannot be tested"
    else:
        stat, pv, k = res
        level = "hard" if pv < thr["hard"] else "soft" if pv < thr["soft"] else "pass"
    leads = {int(kk): (float(v[0]), float(v[1]), float(v[2])) for kk, v in dyn.items() if int(kk) < 0}
    facts = TrendFacts(
        pre_paths=points,
        pre_slope_gap=slope,
        leads=leads,
        leads_stat=stat,
        leads_p=pv,
        leads_k=k,
        leads_level=level,
        leads_how=how,
        composition=_composition(panel, shape),
    )
    return facts, {"dynamic": dyn, "res": list(res) if res else None, "how": how, "error": err}


def _composition(panel: pd.DataFrame, shape: ShapeFacts) -> Composition:
    per = panel.groupby(["time", "treated"])["unit"].nunique().unstack("treated").reindex(columns=[1, 0]).fillna(0).astype(int)
    times = list(per.index)
    if not times:
        return Composition()
    first, last = times[0], times[-1]
    at_first = set(panel.loc[panel["time"] == first, "unit"])
    at_last = set(panel.loc[panel["time"] == last, "unit"])
    every = set(panel["unit"])
    return Composition(
        per_period=[(period_label(t), int(per.loc[t, 1]), int(per.loc[t, 0])) for t in times],
        entries=len(every - at_first),
        exits=len(every - at_last),
        balanced=shape.balanced,
    )


def _leads(panel: pd.DataFrame, controls: list[str], cfg: dict[str, Any], cohorts: int, never_treated: bool):
    """The leads fit and its joint test: (result, every period's coefficient, how it was fitted, error)."""
    vcov = {"CRV1": "unit"} if panel["unit"].nunique() > 2 else "hetero"
    try:
        if cohorts > 1 and not never_treated:
            res, raw = adapter.leads_by_projection(panel, vcov)
            dyn = {str(k): list(v) for k, v in raw.items()}
            how = " (local projections against units not yet treated, one lead per horizon, Bonferroni over the leads)"
        else:
            m = adapter.leads_model(panel, controls, vcov, cohorts=cohorts)
            res = adapter.leads_test(m)
            dyn = {str(k): list(v) for k, v in adapter.dynamic_coefficients(m).items()}
            how = " (two-stage dynamic fit, so units already treated do not stand in as controls)" if cohorts > 1 else ""
    except Exception as ex:
        return None, {}, "", f"could not be computed: {type(ex).__name__}: {str(ex)[:120]}"
    return res, dyn, how, ""


def run_checks(
    panel: pd.DataFrame,
    shape: ShapeFacts,
    controls: list[str],
    contrast_key: str,
    cfg: dict[str, Any],
    *,
    robust_available: bool = True,
    heterogeneity: dict[str, Any] | None = None,
    trends: TrendFacts | None = None,
    trends_raw: dict[str, Any] | None = None,
) -> tuple[list[CheckResult], dict[str, Any]]:
    """The checks, and the facts behind them a figure can draw: the pre-trends fit's coefficient per period relative to the change.
    `robust_available` says whether an estimator built for staggered adoption applies; `heterogeneity` is the cohort test when it
    ran; `trends` is the trends rung, whose leads fit the pre-trends check reads instead of running it again when the design has
    no controls."""
    out: list[CheckResult] = []
    facts: dict[str, Any] = {}
    u = cfg["units"]["min_per_group"]
    later = (shape.units_treated - next(iter(shape.units_by_cohort.values()), 0)) if shape.cohorts > 1 else 0  # units treated after the first cohort
    comparison = shape.units_control + later
    smallest = min(shape.units_treated, comparison)
    out.append(
        CheckResult(
            contrast=contrast_key,
            name="units",
            level="hard" if smallest < u["hard"] else "soft" if smallest < u["soft"] else "pass",
            value=float(smallest),
            threshold=float(u["soft"]),
            detail=f"{shape.units_treated} treated units, {shape.units_control} never treated"
            + (f", and {later} treated later serve as comparison until their own change" if later else ""),
        )
    )
    if shape.units_treated == 1:
        out.append(
            CheckResult(
                contrast=contrast_key,
                name="single_treated_unit",
                level="soft",
                value=1.0,
                detail="one treated unit; clustered standard errors are not meaningful, the randomisation p-value on the estimate is the inference",
            )
        )
    if shape.clusters is not None and shape.kind == "long":
        few = int(cfg["clusters"]["few"]["soft"])
        out.append(
            CheckResult(
                contrast=contrast_key,
                name="few_clusters",
                level="soft" if shape.clusters < few else "pass",
                value=float(shape.clusters),
                threshold=float(few),
                detail=f"{shape.clusters} clusters"
                + ("; below the line where the clustered formula is trusted, so the p-value is resampled" if shape.clusters < few else ""),
            )
        )
    if shape.cohorts > 1 and not shape.never_treated_exists and cfg["never_treated"]["soft_when_none"]:
        out.append(
            CheckResult(
                contrast=contrast_key,
                name="never_treated",
                level="soft",
                value=0.0,
                detail="no unit is never treated; every comparison is with units not yet treated, and the last cohort has no comparison after its own change",
            )
        )
    if shape.periods_pre <= cfg["periods"]["parallel_untestable_when_pre"]:
        out.append(
            CheckResult(
                contrast=contrast_key,
                name="parallel_untestable",
                level="soft",
                value=float(shape.periods_pre),
                detail=f"{shape.periods_pre} pre period; the parallel-trends assumption cannot be tested on this data",
            )
        )
    else:
        r, dyn = _pre_trends(panel, controls, contrast_key, cfg, shape.cohorts, shape.never_treated_exists, trends, trends_raw)
        out.append(r)
        if dyn:
            facts["dynamic"] = dyn
    if trends is not None:
        n = trends.composition.entries + trends.composition.exits
        thr = int(cfg["composition"]["entries_or_exits"]["soft"])
        out.append(
            CheckResult(
                contrast=contrast_key,
                name="composition",
                level="soft" if n >= thr else "pass",
                value=float(n),
                threshold=float(thr),
                detail=f"{trends.composition.entries} units entered after the first period, {trends.composition.exits} left before the last"
                + ("; who is compared changes over the window" if n >= thr else "; the same units throughout"),
            )
        )
    if shape.cohorts > 1:
        out.append(
            CheckResult(
                contrast=contrast_key,
                name="staggered",
                level="pass" if robust_available else ("hard" if cfg["adoption"]["hard_when_no_estimator_applies"] else "soft"),
                value=float(shape.cohorts),
                detail=f"{shape.cohorts} first-treated periods; two-way fixed effects is biased under staggered adoption"
                + (
                    "; the estimators offered compare each cohort only with units not yet treated or never treated"
                    if robust_available
                    else "; no estimator in the catalogue handles it"
                ),
            )
        )
    if heterogeneity is not None:
        thr = float(cfg["cohort_heterogeneity"]["p_value"]["soft"])
        if heterogeneity.get("error"):
            out.append(CheckResult(contrast=contrast_key, name="cohort_heterogeneity", level="soft", detail=f"could not be tested: {heterogeneity['error']}"))
        else:
            p = float(heterogeneity["p"])
            out.append(
                CheckResult(
                    contrast=contrast_key,
                    name="cohort_heterogeneity",
                    level="soft" if p < thr else "pass",
                    value=round(p, 4),
                    threshold=thr,
                    detail=f"test that the cohorts' effects are equal: p = {p:.3g}"
                    + ("; the cohorts moved differently after the change" if p < thr else "; no sign the cohorts differ"),
                )
            )
    return out, facts


def _pre_trends(
    panel: pd.DataFrame,
    controls: list[str],
    contrast_key: str,
    cfg: dict[str, Any],
    cohorts: int,
    never_treated: bool,
    trends: TrendFacts | None = None,
    trends_raw: dict[str, Any] | None = None,
) -> tuple[CheckResult, dict]:
    """The pre-trends check: the trends rung's fit when the design has no controls (the number the comparison rung read); with
    controls, the fit again with them, and the rung's number in the detail so the two can be compared."""
    with_controls = ""
    if trends is not None and trends_raw is not None and not controls:
        res = tuple(trends_raw["res"]) if trends_raw.get("res") else None
        dyn, how, err = trends_raw.get("dynamic") or {}, trends_raw.get("how", ""), trends_raw.get("error", "")
        how = how + " [ladder:trends.leads]"
    else:
        res, dyn, how, err = _leads(panel, controls, cfg, cohorts, never_treated)
        if trends is not None and trends.leads_p is not None:
            with_controls = f"; with the design's controls; without them the trends rung read p = {trends.leads_p:.3g} [ladder:trends.leads]"
    if err:
        return CheckResult(contrast=contrast_key, name="pre_trends", level="soft", detail=err), {}
    if res is None:
        return CheckResult(contrast=contrast_key, name="pre_trends", level="soft", detail="no pre-period coefficients to test"), dyn
    stat, p, k = res
    thr = cfg["pre_trends"]["p_value"]
    level = "hard" if p < thr["hard"] else "soft" if p < thr["soft"] else "pass"
    return CheckResult(
        contrast=contrast_key,
        name="pre_trends",
        level=level,
        value=round(p, 4),
        threshold=float(thr["soft"]),
        detail=f"joint test that the {k} pre-period coefficients are zero: p = {p:.3g}"
        + how
        + with_controls
        + ("; the groups were already moving differently before the change" if level != "pass" else "; no sign of differing pre-trends"),
    ), dyn
