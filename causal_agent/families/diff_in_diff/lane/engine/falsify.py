"""Falsifiable implications of frozen static specifications."""

from __future__ import annotations

import pandas as pd

from causal_agent.common.contracts import Falsification
from causal_agent.families.diff_in_diff.lane.contracts import Design
from causal_agent.families.diff_in_diff.lane.engine.estimators import fit, linear_combination


def run(d: Design, panel: pd.DataFrame, name: str) -> Falsification:
    r = Falsification(design_id=d.id, contrast=d.contrast.key, refuter=name)
    d.assert_integrity()
    p = panel.loc[list(d.row_ids)].copy()
    pre = p.loc[p.post == 0].copy()
    times = sorted(pre.time_index.unique())
    if name != "placebo_timing" or len(times) < 3:
        return r.model_copy(update={"detail": "not enough pre-treatment periods for the declared falsification"})
    cut = times[len(times) // 2]
    pre["treat"] = ((pre.treated == 1) & (pre.time_index >= cut)).astype(float)
    try:
        m = fit(d.formula, pre, d.vcov)
        value, lo, hi, pvalue = linear_combination(m, {"treat": 1.0})
        return r.model_copy(
            update={"new_effect": value, "p_value": pvalue, "passed": lo <= 0 <= hi, "detail": f"fake change at pre-period {cut}: {value:g} [{lo:g}, {hi:g}]"}
        )
    except Exception as ex:
        return r.model_copy(update={"detail": f"falsification unavailable: {type(ex).__name__}: {ex}"})
