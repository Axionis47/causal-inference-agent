"""Sensitivity: how far the estimate moves if a hidden factor of the declared strengths existed. A range, never a verdict."""

from __future__ import annotations

import numpy as np

from causal_agent.common.contracts import Refutation
from causal_agent.families.adjustment.lane.engine.model import SEED
from causal_agent.families.adjustment.lane.knowledge import RefuterEntry


def run(model, bundle, entry: RefuterEntry, contrast_key: str) -> Refutation:
    ide, est_obj = bundle
    np.random.seed(SEED)
    try:
        r = model.refute_estimate(ide, est_obj, method_name=entry.name, show_progress_bar=False, **entry.params)
    except Exception as ex:
        return Refutation(contrast=contrast_key, refuter=entry.name, kind="sensitivity", detail=f"{type(ex).__name__}: {str(ex)[:300]}")
    arr = np.asarray(r.new_effect, dtype=float).ravel()
    lo, hi = (float(np.nanmin(arr)), float(np.nanmax(arr))) if arr.size else (float("nan"), float("nan"))
    return Refutation(
        contrast=contrast_key,
        refuter=entry.name,
        kind="sensitivity",
        range_low=lo,
        range_high=hi,
        detail=f"estimate ranges {lo:.3g} to {hi:.3g} under simulated confounders of the declared strengths",
    )
