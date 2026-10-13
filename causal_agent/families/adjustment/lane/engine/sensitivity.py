"""Sensitivity: how far the estimate moves if a hidden factor of the declared strengths existed. A range, never a verdict."""

from __future__ import annotations

import numpy as np

from causal_agent.common.contracts import Sensitivity
from causal_agent.families.adjustment.lane.engine.model import SEED
from causal_agent.families.adjustment.lane.knowledge import SensitivityEntry


def run(model, bundle, entry: SensitivityEntry, contrast_key: str, design_id: str) -> Sensitivity:
    model, ide, est_obj = bundle
    np.random.seed(SEED)
    try:
        r = model.refute_estimate(ide, est_obj, method_name=entry.name, show_progress_bar=False, **entry.params)
    except Exception as ex:
        return Sensitivity(design_id=design_id, contrast=contrast_key, method=entry.name, error=f"{type(ex).__name__}: {str(ex)[:300]}")
    arr = np.asarray(r.new_effect_array, dtype=float).ravel()
    treatment_strengths = np.atleast_1d(entry.params["effect_strength_on_treatment"])
    outcome_strengths = np.atleast_1d(entry.params["effect_strength_on_outcome"])
    grid = [(float(t), float(y)) for t in treatment_strengths for y in outcome_strengths]
    if len(arr) != len(grid) or not np.isfinite(arr).all():
        return Sensitivity(design_id=design_id, contrast=contrast_key, method=entry.name, error="sensitivity strength grid is incomplete or nonfinite")
    lo, hi = (float(np.nanmin(arr)), float(np.nanmax(arr))) if arr.size else (float("nan"), float("nan"))
    return Sensitivity(
        design_id=design_id,
        contrast=contrast_key,
        method=entry.name,
        range_low=lo,
        range_high=hi,
        points=[{"treatment_strength": t, "outcome_strength": y, "effect": float(value)} for (t, y), value in zip(grid, arr, strict=True)],
        detail=f"estimate ranges {lo:.3g} to {hi:.3g} under simulated confounders of the declared strengths",
    )
