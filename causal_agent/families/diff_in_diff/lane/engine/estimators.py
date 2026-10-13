"""Only frozen specifications reach pyfixest estimation. No adaptive rows or horizons."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from causal_agent.common.contracts import Estimate
from causal_agent.families.diff_in_diff.lane.contracts import Design
from causal_agent.families.diff_in_diff.lane.targets import target_groups

DYNAMIC = re.compile(r"rel_time::(-?\d+)(?:\.0)?(?::treated)?$")
CELL = re.compile(r"rel_time::(-?\d+)(?:\.0)?:first_treated_period::(\d+)")


def formula(controls: tuple[str, ...] | list[str], dynamic: bool = False) -> str:
    rhs = "i(rel_time, treated, ref=-1)" if dynamic else "treat"
    return "y ~ " + rhs + "".join(f" + {c}" for c in controls) + " | unit+time_index"


def fit(spec: str, panel: pd.DataFrame, vcov: str | dict):
    import pyfixest as pf

    return pf.feols(spec, panel.copy(), vcov=vcov)


def _estimate(d: Design, panel: pd.DataFrame, label: str, value: float, lo: float | None, hi: float | None, p: float | None) -> Estimate:
    target = {**d.target, **d.target.get("components", {}).get(label, {}), "component": label}
    target["cohort_calendar"] = [d.target.get("cohort_labels", {}).get(str(g), str(g)) for g in target.get("cohorts", [])]
    modifier, level = label.split(":", 1) if label != "pooled" else (None, None)
    return Estimate(
        design_id=d.id,
        endpoint_id=d.endpoint_id,
        target=target,
        contrast=d.contrast.key,
        method=d.estimator,
        value=value,
        ci_low=lo,
        ci_high=hi,
        p_value=p,
        p_value_source=d.inference,
        n_treated=int(panel.loc[panel.treat == 1, "unit"].nunique()),
        n_control=int(panel.loc[panel.cohort == 0, "unit"].nunique()),
        target_units="att",
        modifier=modifier,
        level=level,
    )


def linear_combination(model: Any, weights: dict[str, float]) -> tuple[float, float, float, float]:
    missing = set(weights) - set(model.coef().index)
    if missing:
        raise ValueError(f"frozen target coefficients are unavailable: {sorted(missing)}")
    vector = np.array([weights.get(n, 0.0) for n in model.coef().index])
    value = float(vector @ model.coef().to_numpy())
    covariance = np.asarray(model._vcov)
    variance = float(vector @ covariance @ vector)
    if not np.isfinite(variance) or variance < -1e-10:
        raise ValueError("the target covariance is invalid")
    se = float(np.sqrt(max(variance, 0.0)))
    df = float(model._df_t)
    if not np.isfinite(df) or df <= 0:
        raise ValueError("native covariance inference has invalid degrees of freedom")
    critical = float(student_t.ppf(0.975, df))
    return value, value - critical * se, value + critical * se, float(2 * student_t.sf(abs(value / se), df)) if se else 1.0 if value == 0 else 0.0


def run(d: Design, panel: pd.DataFrame) -> list[Estimate]:
    """Fit exactly the frozen rows and specifications; failures are durable records."""
    import pyfixest as pf

    d.assert_integrity()
    p = panel.loc[list(d.row_ids)].copy().sort_values(["unit", "time_index"]).reset_index(drop=True)
    try:
        if d.engine == "lpdid":
            return _projection(d, p)
        if d.engine == "feols":
            m = fit(d.formula, p, d.vcov)
        elif d.engine == "did2s":
            for g in sorted(p.loc[p.treat == 1, "cohort"].unique()):
                p[f"cohort_treat_{int(g)}"] = ((p.cohort == g) & (p.treat == 1)).astype(int)
            m = pf.did2s(p, yname="y", first_stage=d.first_stage, second_stage=d.second_stage, treatment="treat", cluster=d.cluster_column)
        else:
            if d.controls:
                raise ValueError("saturated engine does not implement covariate adjustment")
            m = pf.event_study(p, yname="y", idname="unit", tname="time_index", gname="cohort", estimator="saturated", cluster=d.cluster_column)
        groups = target_groups(p, d.target)
        out = []
        for label, mask in groups:
            if not mask.any():
                raise ValueError(f"no admitted target observations for {label}")
            if d.estimator in {"twfe_static", "did2s"}:
                w = {"treat": 1.0}
            elif d.estimator == "did2s_cohort":
                g = int(label.split(":")[1])
                w = {f"cohort_treat_{g}": 1.0}
            elif d.engine == "saturated":
                cells = p.loc[mask].groupby(["cohort", "rel_time"]).size()
                total = int(cells.sum())
                w = {}
                for (g, h), count in cells.items():
                    name = next((n for n in m.coef().index if (hit := CELL.search(n)) and int(hit.group(1)) == int(h) and int(hit.group(2)) == int(g)), None)
                    if name is None:
                        raise ValueError(f"frozen cohort/horizon cell {g}/{h} missing")
                    w[name] = int(count) / total
            else:
                h = int(label.split(":")[1])
                name = next((n for n in m.coef().index if (hit := DYNAMIC.search(n)) and int(hit.group(1)) == h), None)
                if name is None:
                    raise ValueError(f"frozen horizon {h} missing")
                w = {name: 1.0}
            value, lo, hi, pv = linear_combination(m, w)
            out.append(_estimate(d, p.loc[mask | (p.treat == 0)], label, value, lo, hi, pv))
        if d.engine == "feols" and d.inference.startswith("wild_"):
            for e in out:
                if e.modifier is None:
                    boot = m.wildboottest(
                        reps=999, cluster=d.cluster_column, param="treat", weights_type="webb" if d.inference == "wild_webb" else "rademacher", seed=7
                    )
                    e.p_value = float(boot["Pr(>|t|)"])
                    e.p_value_source = d.inference
        return out
    except Exception as ex:
        return [
            Estimate(
                design_id=d.id,
                endpoint_id=d.endpoint_id,
                target=d.target,
                contrast=d.contrast.key,
                method=d.estimator,
                target_units="att",
                error=f"{type(ex).__name__}: {str(ex)[:350]}",
            )
        ]


def _projection(d: Design, p: pd.DataFrame) -> list[Estimate]:
    import pyfixest as pf

    mode = d.target["aggregation"]
    horizons = d.target["horizons"]
    H = max(horizons)
    m = pf.lpdid(p.copy(), yname="y", idname="unit", tname="time_index", gname="cohort", vcov=d.vcov, pre_window=-2, post_window=H, att=mode == "pooled")
    tidy = m.tidy()
    out = []
    if mode == "pooled":
        r = tidy.loc["treat_diff"]
        return [_estimate(d, p, "pooled", float(r["Estimate"]), float(r["2.5%"]), float(r["97.5%"]), float(r["Pr(>|t|)"]))]
    for h in horizons:
        r = tidy.loc[f"time_to_treatment::{h}"]
        out.append(_estimate(d, p, f"horizon:{h}", float(r["Estimate"]), float(r["2.5%"]), float(r["97.5%"]), float(r["Pr(>|t|)"])))
    return out
