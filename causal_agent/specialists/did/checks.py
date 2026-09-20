"""Design checks on the canonical panel: facts, flagged against checks.yaml. No model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from causal_agent.common.contracts import CheckResult
from causal_agent.specialists.did import adapter
from causal_agent.specialists.did.contracts import ShapeFacts


def run_checks(panel: pd.DataFrame, shape: ShapeFacts, controls: list[str], contrast_key: str, cfg: dict[str, Any]) -> tuple[list[CheckResult], dict[str, Any]]:
    """The checks, and the facts behind them a figure can draw: the pre-trends fit's coefficient per period relative to the change."""
    out: list[CheckResult] = []
    facts: dict[str, Any] = {}
    u = cfg["units"]["min_per_group"]
    smallest = min(shape.units_treated, shape.units_control)
    out.append(CheckResult(contrast=contrast_key, name="units", level="hard" if smallest < u["hard"] else "soft" if smallest < u["soft"] else "pass",
                           value=float(smallest), threshold=float(u["soft"]), detail=f"{shape.units_treated} treated units, {shape.units_control} control"))
    if shape.units_treated == 1:
        out.append(CheckResult(contrast=contrast_key, name="single_treated_unit", level="soft", value=1.0,
                               detail="one treated unit; clustered standard errors are not meaningful, the placebo-group p-value is the inference"))
    if shape.periods_pre <= cfg["periods"]["parallel_untestable_when_pre"]:
        out.append(CheckResult(contrast=contrast_key, name="parallel_untestable", level="soft", value=float(shape.periods_pre),
                               detail=f"{shape.periods_pre} pre period; the parallel-trends assumption cannot be tested on this data"))
    else:
        r, dyn = _pre_trends(panel, controls, contrast_key, cfg)
        out.append(r)
        if dyn:
            facts["dynamic"] = dyn
    if shape.cohorts > 1:
        out.append(CheckResult(contrast=contrast_key, name="staggered", level="hard" if cfg["adoption"]["staggered_is_hard"] else "soft", value=float(shape.cohorts),
                               detail=f"{shape.cohorts} first-treated periods; two-way fixed effects can be biased under staggered adoption"))
    return out, facts


def _pre_trends(panel: pd.DataFrame, controls: list[str], contrast_key: str, cfg: dict[str, Any]) -> tuple[CheckResult, dict]:
    formula = "y ~ i(rel_time, treated, ref=-1)" + (" + " + " + ".join(controls) if controls else "") + " | unit+time"
    vcov = {"CRV1": "unit"} if panel["unit"].nunique() > 2 else "hetero"
    try:
        m = adapter.fit(formula, panel, vcov)
        res = adapter.leads_test(m)
        dyn = {str(k): list(v) for k, v in adapter.dynamic_coefficients(m).items()}
    except Exception as ex:
        return CheckResult(contrast=contrast_key, name="pre_trends", level="soft", detail=f"could not be computed: {type(ex).__name__}: {str(ex)[:120]}"), {}
    if res is None:
        return CheckResult(contrast=contrast_key, name="pre_trends", level="soft", detail="no pre-period coefficients to test"), dyn
    stat, p, k = res
    thr = cfg["pre_trends"]["p_value"]
    level = "hard" if p < thr["hard"] else "soft" if p < thr["soft"] else "pass"
    return CheckResult(contrast=contrast_key, name="pre_trends", level=level, value=round(p, 4), threshold=float(thr["soft"]),
                       detail=f"joint test that the {k} pre-period coefficients are zero: p = {p:.3g}" +
                              ("; the groups were already moving differently before the change" if level != "pass" else "; no sign of differing pre-trends")), dyn
