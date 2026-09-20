"""The only file that imports pyfixest. Reads a frozen Design and the canonical panel, returns artifacts.

One fit function for every formula shape. Inference is a vcov passed through. Placebos refit the same
formula on a perturbed panel. Failures come back as facts on the artifact, never as exceptions.
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from causal_agent.common.contracts import Estimate, Refutation
from causal_agent.specialists.did.knowledge import EstimatorEntry, PlaceboEntry

logging.getLogger("pyfixest").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="pyfixest")

SEED = 7
COEF = "treat"
DYNAMIC = re.compile(r"rel_time::(-?\d+):treated")


def formula_for(entry: EstimatorEntry, controls: list[str]) -> str:
    """Fill {controls} from the entry's mode. Never sees a raw column name from the model."""
    if not controls or entry.controls_mode == "none":
        return entry.formula.replace("{controls}", "")
    if entry.controls_mode == "csw0" and len(controls) >= 2:  # pyfixest's stepwise operators need at least two terms
        return entry.formula.replace("{controls}", " + csw0(" + ", ".join(controls) + ")")
    return entry.formula.replace("{controls}", " + " + " + ".join(controls))


def fit(formula: str, panel: pd.DataFrame, vcov: Any):
    import pyfixest as pf

    return pf.feols(formula, panel, vcov=vcov)


def estimate(entry: EstimatorEntry, formula: str, panel: pd.DataFrame, vcov: Any, contrast_key: str, target_units: str, *, secondary: bool = False) -> tuple[list[Estimate], Any]:
    """Returns one Estimate per fitted model (csw0 yields several: no controls, then each added) and the primary fit object
    for follow-ups. Under csw0 the primary is the full formula the design advertises, the last model; the steps are secondary,
    labelled by how many controls they carry."""
    n_t = int(panel.loc[panel["treated"] == 1, "unit"].nunique())
    n_c = int(panel.loc[panel["treated"] == 0, "unit"].nunique())
    try:
        res = fit(formula, panel, vcov)
        models = res.to_list() if hasattr(res, "to_list") else [res]
        stepwise = entry.controls_mode == "csw0" and len(models) > 1
        primary_i = len(models) - 1 if stepwise else 0
        out: list[Estimate] = []
        for i, m in enumerate(models):
            label = entry.name if i == primary_i else f"{entry.name}+{i}"
            if COEF in m.coef().index:
                lo, hi = (float(x) for x in m.confint().loc[COEF].to_numpy())
                out.append(Estimate(contrast=contrast_key, method=label, value=float(m.coef()[COEF]), ci_low=lo, ci_high=hi,
                                    n_treated=n_t, n_control=n_c, target_units=target_units, secondary=secondary or i != primary_i))
            else:  # dynamic model: the effect is the mean of post-period coefficients; each period is reported separately
                names = [n for n in m.coef().index if DYNAMIC.search(n)]
                lags = [n for n in names if int(DYNAMIC.search(n).group(1)) >= 0]
                if lags:
                    vals = m.coef()[lags]
                    out.append(Estimate(contrast=contrast_key, method=label, value=float(vals.mean()), n_treated=n_t, n_control=n_c,
                                        target_units=target_units, secondary=True))
        return out, models[primary_i]
    except Exception as ex:
        return [Estimate(contrast=contrast_key, method=entry.name, n_treated=n_t, n_control=n_c, target_units=target_units,
                         secondary=secondary, error=f"{type(ex).__name__}: {str(ex)[:300]}")], None


def dynamic_coefficients(model) -> dict[int, tuple[float, float, float]]:
    """rel_time -> (estimate, ci_low, ci_high) from a dynamic fit."""
    out = {}
    ci = model.confint()
    for name, val in model.coef().items():
        m = DYNAMIC.search(name)
        if m:
            k = int(m.group(1))
            out[k] = (float(val), float(ci.loc[name].iloc[0]), float(ci.loc[name].iloc[1]))
    return out


def leads_test(model) -> tuple[float, float, int] | None:
    """Joint test that every pre-period coefficient is zero. (statistic, p_value, number of leads) or None."""
    names = list(model.coef().index)
    leads = [n for n in names if DYNAMIC.search(n) and int(DYNAMIC.search(n).group(1)) < 0]
    if not leads:
        return None
    R = np.zeros((len(leads), len(names)))
    for i, n in enumerate(leads):
        R[i, names.index(n)] = 1
    w = model.wald_test(R=R, q=np.zeros(len(leads)), distribution="chi2")
    return float(w["statistic"]), float(w["pvalue"]), len(leads)


def wild_bootstrap(model, reps: int, seed: int) -> float | None:
    try:
        r = model.wildboottest(param=COEF, reps=reps, seed=seed)
        p = r.get("Pr(>|t|)") if hasattr(r, "get") else None
        return float(p) if p is not None else None
    except Exception:
        return None


def placebo_group(formula: str, panel: pd.DataFrame, observed: float, entry: PlaceboEntry, contrast_key: str) -> tuple[Refutation, list[float]]:
    """Reassign the treated label across units at random and refit. p = share of |placebo| >= |observed|. Every placebo
    effect comes back too, so the spread can be drawn."""
    rng = np.random.default_rng(SEED)
    labels = panel.groupby("unit")["treated"].first()
    draws = int(entry.params.get("draws", 200))
    effects: list[float] = []
    for _ in range(draws):
        perm = pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)
        p2 = panel.copy()
        p2["treated"] = p2["unit"].map(perm).astype(int)
        p2["treat"] = (p2["treated"] * p2["post"]).astype(float)
        try:
            m = fit(formula, p2, "iid")
            m = m.to_list()[0] if hasattr(m, "to_list") else m
            effects.append(float(m.coef()[COEF]))
        except Exception:
            continue
    if not effects:
        return Refutation(contrast=contrast_key, refuter=entry.name, kind="falsification", detail="no placebo fit succeeded"), []
    p = float(np.mean(np.abs(effects) >= abs(observed)))
    passed = p < float(entry.pass_when.get("p_value_lt", 0.05))
    return Refutation(contrast=contrast_key, refuter=entry.name, kind="falsification", new_effect=float(np.mean(effects)), p_value=p, passed=passed,
                      detail=f"{len(effects)} reassignments; share with an effect at least as large: {p:.2f}" + (" (pass)" if passed else " (FAIL: the observed effect is not unusual)")), effects


def placebo_timing(formula: str, panel: pd.DataFrame, entry: PlaceboEntry, contrast_key: str) -> Refutation:
    """Pre-period rows only, with a fake change in the middle of the pre window. The effect should be about zero."""
    pre = panel[panel["post"] == 0].copy()
    times = sorted(pre["time"].unique())
    if len(times) < 3:
        return Refutation(contrast=contrast_key, refuter=entry.name, kind="falsification", detail="fewer than three pre periods")
    cut = times[len(times) // 2]
    pre["post"] = (pre["time"] >= cut).astype(int)
    pre["treat"] = (pre["treated"] * pre["post"]).astype(float)
    pre["rel_time"] = pre["time"].map({v: i for i, v in enumerate(times)}) - times.index(cut)
    try:
        m = fit(formula, pre, {"CRV1": "unit"} if pre["unit"].nunique() > 2 else "hetero")
        m = m.to_list()[0] if hasattr(m, "to_list") else m
        val = float(m.coef()[COEF])
        lo, hi = (float(x) for x in m.confint().loc[COEF].to_numpy())
    except Exception as ex:
        return Refutation(contrast=contrast_key, refuter=entry.name, kind="falsification", detail=f"{type(ex).__name__}: {str(ex)[:200]}")
    passed = lo <= 0 <= hi
    return Refutation(contrast=contrast_key, refuter=entry.name, kind="falsification", new_effect=val, passed=passed,
                      detail=f"fake change at {cut:g}: effect {val:.3g} [{lo:.3g}, {hi:.3g}]" + (" (pass)" if passed else " (FAIL: a pre-period 'effect' that is not zero)"))
