"""Falsification: move something that should not matter, or should, and see whether the estimate reacts. A verdict, by the rule the
entry declares."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from causal_agent.common.contracts import Estimate, Refutation
from causal_agent.families.adjustment.lane.engine.model import SEED
from causal_agent.families.adjustment.lane.knowledge import RefuterEntry


def run(model, bundle, est: Estimate, entry: RefuterEntry, contrast_key: str) -> Refutation:
    ide, est_obj = bundle
    np.random.seed(SEED)  # reproducible draws without DoWhy's per-simulation reseed (see the catalogue)
    try:
        r = model.refute_estimate(ide, est_obj, method_name=entry.name, show_progress_bar=False, **entry.params)
    except Exception as ex:
        return Refutation(contrast=contrast_key, refuter=entry.name, kind="falsification", detail=f"{type(ex).__name__}: {str(ex)[:300]}")
    new = float(r.new_effect)
    res = r.refutation_result or {}
    p = res.get("p_value")
    p = None if p is None or (isinstance(p, float) and math.isnan(float(p))) else float(p)
    passed = _passes(entry.pass_when, est.value, new, p)
    return Refutation(
        contrast=contrast_key,
        refuter=entry.name,
        kind="falsification",
        new_effect=new,
        p_value=p,
        passed=passed,
        detail=f"new effect {new:.3g}" + (f", p={p:.2f}" if p is not None else ", p not computable") + (" (pass)" if passed else " (FAIL)"),
    )


def _passes(pass_when: dict[str, Any], value: float | None, new: float, p: float | None) -> bool:
    if value is None:
        return False
    thr = pass_when.get("p_value_gt")
    if p is not None and thr is not None:
        return p > thr
    scale = abs(value) if abs(value) > 1e-9 else 1.0
    if pass_when.get("new_effect_near_zero"):
        return abs(new) <= 0.1 * scale
    rel = pass_when.get("or_relative_change_lt")
    if rel is not None:
        return abs(new - value) <= rel * scale
    return False
