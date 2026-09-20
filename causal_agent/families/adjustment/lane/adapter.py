"""The only file that imports DoWhy. Reads a frozen Design, makes the calls, returns artifacts.

No branching on method: every estimator is a name plus a dict, every refuter the same.
Failures come back as facts on the artifact (Estimate.error), never as exceptions.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # the sensitivity refuter draws; nothing here has a screen

from causal_agent.common.contracts import Contrast, Estimate, Refutation
from causal_agent.families.adjustment.lane.contracts import Estimand, Graph
from causal_agent.families.adjustment.lane.knowledge import EstimatorEntry, RefuterEntry

logging.getLogger("dowhy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="dowhy")
warnings.filterwarnings("ignore", module="sklearn")
warnings.filterwarnings("ignore", module="statsmodels")

TREATED = "treated"  # the 0/1 column every contrast table carries
HIDDEN = "unobserved"  # the node that stands for a factor the person says exists and the file does not hold
SEED = 7


def contrast_table(table: pd.DataFrame, treatment: str, contrast: Contrast) -> pd.DataFrame:
    """Rows at the two levels only, with a 0/1 `treated` column. DoWhy always sees a binary treatment."""
    col = table[treatment].astype(str)
    sub = table[col.isin([str(contrast.control), str(contrast.treated)])].copy()
    flag = (col.loc[sub.index] == str(contrast.treated)).astype(int)
    if treatment != TREATED:  # a file whose treatment column is itself named "treated" keeps its place
        sub = sub.drop(columns=[treatment])
    sub[TREATED] = flag
    return sub


def build_model(table: pd.DataFrame, graph: Graph, outcome: str):
    from dowhy import CausalModel

    g = graph.to_networkx()
    g = _relabel(g, graph.treatment, TREATED)
    return CausalModel(data=table, treatment=TREATED, outcome=outcome, graph=g)


def identify(model) -> Estimand:
    """Every road DoWhy finds on the graph. The design takes the first open one in the order backdoor, frontdoor, iv; the
    estimator pick may take another, and the frozen design records which."""
    ide = model.identify_effect(proceed_when_unidentifiable=True)
    ests = getattr(ide, "estimands", {}) or {}
    roads = [k for k in ("backdoor", "frontdoor", "iv") if ests.get(k) is not None]
    if not roads:
        return Estimand(kind="none", dowhy_text=str(ide))
    alts = {k: list(v) for k, v in (getattr(ide, "backdoor_variables", {}) or {}).items() if k != "backdoor"}
    return Estimand(
        kind=roads[0],
        roads=roads,
        adjustment_set=list(ide.get_backdoor_variables()) if "backdoor" in roads else [],
        instruments=[str(v) for v in (ide.get_instrumental_variables() or [])] if "iv" in roads else [],
        frontdoor_set=[str(v) for v in (ide.get_frontdoor_variables() or [])] if "frontdoor" in roads else [],
        alternatives=alts,
        dowhy_text=str(ide),
    )


def estimate(model, entry: EstimatorEntry, contrast_key: str, target_units: str, *, secondary: bool = False) -> tuple[Estimate, Any]:
    """Returns the artifact and DoWhy's estimate object (needed by refuters; never stored in state)."""
    table = model._data
    n_t, n_c = int((table[TREATED] == 1).sum()), int((table[TREATED] == 0).sum())
    ide = model.identify_effect(proceed_when_unidentifiable=True)
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
            (ide, est),  # estimate_effect stamps the identifier method on `ide`; refuters need that same object
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


def refute(model, bundle, est: Estimate, entry: RefuterEntry, contrast_key: str) -> Refutation:
    ide, est_obj = bundle
    np.random.seed(SEED)  # reproducible draws without DoWhy's per-simulation reseed (see refuters.yaml)
    try:
        r = model.refute_estimate(ide, est_obj, method_name=entry.name, show_progress_bar=False, **entry.params)
    except Exception as ex:
        return Refutation(contrast=contrast_key, refuter=entry.name, kind=entry.kind, detail=f"{type(ex).__name__}: {str(ex)[:300]}")
    if entry.kind == "sensitivity":
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


# ------------------------------------------------------------------ helpers


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


def _params(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    if out.get("glm_family") == "binomial":
        import statsmodels.api as sm

        out["glm_family"] = sm.families.Binomial()
    return out


def _relabel(g, old: str, new: str):
    import networkx as nx

    return nx.relabel_nodes(g, {old: new}) if old in g else g
