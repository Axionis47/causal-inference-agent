"""Design checks on the canonical panel: facts, flagged against checks.yaml. No model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from causal_agent.common.contracts import CheckResult
from causal_agent.families.diff_in_diff.lane import adapter
from causal_agent.families.diff_in_diff.lane.contracts import ShapeFacts


def run_checks(
    panel: pd.DataFrame,
    shape: ShapeFacts,
    controls: list[str],
    contrast_key: str,
    cfg: dict[str, Any],
    *,
    robust_available: bool = True,
    heterogeneity: dict[str, Any] | None = None,
) -> tuple[list[CheckResult], dict[str, Any]]:
    """The checks, and the facts behind them a figure can draw: the pre-trends fit's coefficient per period relative to the change.
    `robust_available` says whether an estimator built for staggered adoption applies; `heterogeneity` is the cohort test when it ran."""
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
                detail="one treated unit; clustered standard errors are not meaningful, the placebo-group p-value is the inference",
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
        r, dyn = _pre_trends(panel, controls, contrast_key, cfg, shape.cohorts, shape.never_treated_exists)
        out.append(r)
        if dyn:
            facts["dynamic"] = dyn
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
    panel: pd.DataFrame, controls: list[str], contrast_key: str, cfg: dict[str, Any], cohorts: int, never_treated: bool
) -> tuple[CheckResult, dict]:
    vcov = {"CRV1": "unit"} if panel["unit"].nunique() > 2 else "hetero"
    how = ""
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
        return CheckResult(contrast=contrast_key, name="pre_trends", level="soft", detail=f"could not be computed: {type(ex).__name__}: {str(ex)[:120]}"), {}
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
        + ("; the groups were already moving differently before the change" if level != "pass" else "; no sign of differing pre-trends"),
    ), dyn
