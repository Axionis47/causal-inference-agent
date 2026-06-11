"""Matching lane: 1-NN propensity matching with replacement, ATT.

Standard error via a matched-sample bootstrap that re-estimates the
propensity model inside every draw. Balance before/after rides as a
table artifact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis_v2.spec import CausalSpec, EffectEstimate, EstimateResult, MethodLane, MethodPlan

from .common import (
    LaneArtifact,
    LaneInputError,
    LaneOutcome,
    binary_01,
    ci_from,
    effects_table,
    numeric_frame,
    safe_float,
    summary_markdown,
)

_BOOTSTRAP_DRAWS = 200


def _att_once(y: np.ndarray, t: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    p = (
        LogisticRegression(max_iter=300)
        .fit(x, t)
        .predict_proba(x)[:, 1]
        .reshape(-1, 1)
    )
    treated, control = p[t == 1], p[t == 0]
    nn = NearestNeighbors(n_neighbors=1).fit(control)
    _, idx = nn.kneighbors(treated)
    matched = y[t == 0][idx.ravel()]
    return float(np.mean(y[t == 1] - matched)), idx.ravel()


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def run(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> LaneOutcome:
    lane = "matching"
    if plan.treatment is None:
        raise LaneInputError(f"{lane}: needs a binary treatment column")
    if not plan.covariates:
        raise LaneInputError(f"{lane}: needs covariates to match on")
    data = numeric_frame(frame, [plan.outcome, plan.treatment, *plan.covariates], lane)
    t = binary_01(data[plan.treatment], "treatment", lane).to_numpy()
    if t.sum() < 10 or (1 - t).sum() < 10:
        raise LaneInputError(f"{lane}: needs at least 10 units in each arm")
    y = data[plan.outcome].to_numpy(dtype=float)
    x = data[plan.covariates].to_numpy(dtype=float)

    att, matched_idx = _att_once(y, t, x)
    rng = np.random.default_rng(7)
    draws = []
    n = len(y)
    for _ in range(_BOOTSTRAP_DRAWS):
        pick = rng.choice(n, size=n, replace=True)
        if t[pick].sum() < 5 or (1 - t[pick]).sum() < 5:
            continue
        try:
            draws.append(_att_once(y[pick], t[pick], x[pick])[0])
        except Exception:
            continue
    se = float(np.std(draws)) if len(draws) > 50 else None
    lo, hi = ci_from(att, se) if se else (None, None)

    balance_rows = []
    control_pool = x[t == 0][matched_idx]
    for j, cov in enumerate(plan.covariates):
        balance_rows.append(
            {
                "covariate": cov,
                "smd_before": round(_smd(x[t == 1][:, j], x[t == 0][:, j]), 4),
                "smd_after": round(_smd(x[t == 1][:, j], control_pool[:, j]), 4),
            }
        )
    balance = pd.DataFrame(balance_rows)
    worst_after = float(balance["smd_after"].abs().max())
    warnings = []
    if worst_after > 0.1:
        warnings.append(f"residual imbalance after matching (max |SMD| {worst_after:.2f})")

    result = EstimateResult(
        lane=MethodLane.MATCHING,
        estimator=plan.estimator,
        effects=[
            EffectEstimate(
                estimand="att",
                estimate=safe_float(att),
                std_error=se,
                ci_lower=lo,
                ci_upper=hi,
                interpretation="treated vs matched-control difference "
                f"({int(t.sum())} treated, 1-NN with replacement, "
                f"{len(draws)} bootstrap refits)",
            )
        ],
        n_rows_used=len(data),
        outcome=plan.outcome,
        treatment=plan.treatment,
        covariates_used=list(plan.covariates),
        warnings=warnings,
    )
    summary = summary_markdown(
        "Propensity matching",
        [
            f"{int(t.sum())} treated matched into {int((1 - t).sum())} controls",
            f"ATT {att:.4g}" + (f" (bootstrap se {se:.3g})" if se else ""),
            f"max |SMD| after matching: {worst_after:.3f}",
        ],
    )
    return LaneOutcome(
        result=result,
        artifacts=[
            LaneArtifact("model_summary", "markdown", "Matching summary", summary),
            LaneArtifact("effects", "table", "Estimated effects", effects_table(result)),
            LaneArtifact("balance_after_matching", "table", "Balance before/after", balance),
        ],
        warnings=warnings,
    )
