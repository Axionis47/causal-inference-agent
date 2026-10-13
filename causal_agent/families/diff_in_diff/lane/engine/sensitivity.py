"""Robustness ranges, kept separate from falsification verdicts."""

from __future__ import annotations

import pandas as pd

from causal_agent.common.contracts import Sensitivity
from causal_agent.families.diff_in_diff.lane.contracts import Design
from causal_agent.families.diff_in_diff.lane.engine.estimators import fit, formula, linear_combination


def run(d: Design, panel: pd.DataFrame, name: str) -> Sensitivity:
    d.assert_integrity()
    p = panel.loc[list(d.row_ids)].copy()
    spec = formula(d.controls) if name == "twfe_naive" else d.formula
    subsets = [("with unit trends", p)]
    if name == "unit_trends":
        spec = spec.replace("| unit+time_index", "| unit[time_index]+time_index")
    elif name == "leave_one_out":
        subsets = [(f"without {u}", p.loc[p.unit != u]) for u in sorted(p.loc[p.treated == 1, "unit"].unique())]
    else:
        subsets = [("naive staggered TWFE", p)]
    points = []
    for label, rows in subsets:
        try:
            d.assert_integrity()
            m = fit(spec, rows, d.vcov)
            value, lo, hi, _ = linear_combination(m, {"treat": 1.0})
            points.append(dict(label=label, value=value, lo=lo, hi=hi))
        except Exception as ex:
            points.append(dict(label=label, error=f"{type(ex).__name__}: {str(ex)[:150]}"))
    values = [x["value"] for x in points if "value" in x]
    return Sensitivity(
        design_id=d.id,
        contrast=d.contrast.key,
        method=name,
        points=points,
        range_low=min(values) if values else None,
        range_high=max(values) if values else None,
        target={**d.target, "different_estimand": name == "twfe_naive"},
        detail="Robustness refits; no pass/fail verdict. Targets may differ from the primary.",
        error=None if len(values) == len(points) else "one or more refits failed",
    )
