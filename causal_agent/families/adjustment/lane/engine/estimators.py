"""The estimators: a frozen design's entry run on the causal model. The engine identifies again on the frozen graph; its set for
the entry's road must be the one the design froze, or nothing runs and the mismatch is the fact."""

from __future__ import annotations

from typing import Any

import numpy as np

from causal_agent.common.contracts import Estimate
from causal_agent.families.adjustment.lane.contracts import Estimand
from causal_agent.families.adjustment.lane.engine.model import TREATED, frozen_set, road_set
from causal_agent.families.adjustment.lane.knowledge import EstimatorEntry


def estimate(model, entry: EstimatorEntry, contrast_key: str, target_units: str, *, frozen: Estimand, secondary: bool = False) -> tuple[Estimate, Any]:
    """Returns the record and DoWhy's estimate object (the falsifications and the sensitivity analyses need it; never stored in
    state)."""
    table = model._data
    n_t, n_c = int((table[TREATED] == 1).sum()), int((table[TREATED] == 0).sum())
    if target_units not in entry.targets:
        why = f"design: {entry.name} does not serve the target {target_units}; it serves {', '.join(entry.targets)}"
        return Estimate(contrast=contrast_key, method=entry.name, n_treated=n_t, n_control=n_c, target_units=target_units, secondary=secondary, error=why), None
    ide = model.identify_effect(proceed_when_unidentifiable=True)
    found, want = road_set(ide, entry.estimand), frozen_set(frozen, entry.estimand)
    if found != want:
        why = (
            f"design: the engine opens no {entry.estimand} road on the frozen graph"
            if found is None
            else f"design: the engine identified {sorted(found)} for the {entry.estimand} road and the design froze {sorted(want)}"
        )
        return Estimate(contrast=contrast_key, method=entry.name, n_treated=n_t, n_control=n_c, target_units=target_units, secondary=secondary, error=why), None
    try:
        est = model.estimate_effect(
            ide,
            method_name=entry.dowhy,
            control_value=0,
            treatment_value=1,
            target_units=target_units,
            confidence_intervals=True,
            effect_modifiers=[],
            method_params=_params(entry.params),
        )
        ci = est.get_confidence_intervals()
        lo, hi = (None, None)
        if ci is not None:
            arr = np.asarray(ci, dtype=float).ravel()
            if arr.size >= 2:
                lo, hi = float(arr[0]), float(arr[1])
        return (
            Estimate(
                contrast=contrast_key,
                method=entry.name,
                value=float(est.value),
                ci_low=lo,
                ci_high=hi,
                n_treated=n_t,
                n_control=n_c,
                target_units=target_units,
                secondary=secondary,
            ),
            (ide, est),  # estimate_effect stamps the identifier method on `ide`; the runs after it need that same object
        )
    except Exception as ex:  # a fit failure is a fact
        return (
            Estimate(
                contrast=contrast_key,
                method=entry.name,
                n_treated=n_t,
                n_control=n_c,
                target_units=target_units,
                secondary=secondary,
                error=f"{type(ex).__name__}: {str(ex)[:300]}",
            ),
            None,
        )


def _params(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    if out.get("glm_family") == "binomial":
        import statsmodels.api as sm

        out["glm_family"] = sm.families.Binomial()
    return out
