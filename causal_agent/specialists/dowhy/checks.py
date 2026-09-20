"""Design checks: facts about the comparison before any estimate. Thresholds come from checks.yaml."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from causal_agent.common.contracts import CheckResult

TREATED = "treated"


def outcome_kind(s: pd.Series) -> str | None:
    """continuous, binary, or None when the lane cannot use it."""
    if pd.api.types.is_bool_dtype(s):
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        return "binary" if s.dropna().nunique() == 2 else "continuous"
    if s.dropna().nunique() == 2:
        return "binary"
    return None


def run_checks(table: pd.DataFrame, adjustment_set: list[str], contrast_key: str, cfg: dict[str, Any]) -> tuple[list[CheckResult], dict[str, Any]]:
    """The checks, and the facts behind them a figure can draw: the balance of every adjustment column before and after
    weighting on the propensity, and the propensity's separation and common support."""
    out: list[CheckResult] = []
    facts: dict[str, Any] = {"balance": {}, "propensity": {}}
    t = table[TREATED] == 1
    n_t, n_c = int(t.sum()), int((~t).sum())
    thr = float(cfg["arms"]["min_per_arm"]["hard"])
    out.append(CheckResult(contrast=contrast_key, name="arms", level="hard" if min(n_t, n_c) < thr else "pass",
                           value=float(min(n_t, n_c)), threshold=thr, detail=f"{n_t} treated, {n_c} control"))
    if not adjustment_set:
        out.append(CheckResult(contrast=contrast_key, name="overlap", level="pass", detail="nothing to adjust for; overlap holds by construction"))
        return out, facts

    X = _design_matrix(table[adjustment_set])
    scores, auc = _propensity(X, t.astype(int).to_numpy())
    weights = np.where(t.to_numpy(), 1.0 / np.clip(scores, 1e-3, 1 - 1e-3), 1.0 / np.clip(1 - scores, 1e-3, 1 - 1e-3))
    facts["propensity"] = {"auc": round(float(auc), 4), "min_treated": round(float(scores[t].min()), 4), "max_treated": round(float(scores[t].max()), 4),
                           "min_control": round(float(scores[~t].min()), 4), "max_control": round(float(scores[~t].max()), 4)}
    lo = max(scores[t].min(), scores[~t].min())
    hi = min(scores[t].max(), scores[~t].max())
    share = float(((scores >= lo) & (scores <= hi)).mean()) if hi >= lo else 0.0
    o = cfg["overlap"]
    level = "hard" if share < o["common_support_share"]["hard"] else "soft" if share < o["common_support_share"]["soft"] else "pass"
    out.append(CheckResult(contrast=contrast_key, name="overlap", level=level, value=round(share, 3), threshold=o["common_support_share"]["soft"],
                           detail=f"{share:.0%} of rows inside the score range both arms cover ({lo:.2f} to {hi:.2f})"))
    auc_thr = float(o["max_auc"]["hard"])
    out.append(CheckResult(contrast=contrast_key, name="separation", level="hard" if auc >= auc_thr else "pass", value=round(auc, 3), threshold=auc_thr,
                           detail=f"a score model tells the arms apart with AUC {auc:.2f}" + ("; treatment is nearly a function of the adjustment set" if auc >= auc_thr else "")))

    b = cfg["balance"]["smd"]
    for col in adjustment_set:
        smd = _smd(table[col], t)
        after = _smd(table[col], t, weights)
        facts["balance"][col] = {"before": round(smd, 4), "after": round(after, 4)}
        level = "hard" if smd >= b["hard"] else "soft" if smd >= b["soft"] else "pass"
        out.append(CheckResult(contrast=contrast_key, name=f"balance.{col}", level=level, value=round(smd, 3), threshold=b["soft"],
                               detail=f"standardised mean difference {smd:.2f} between arms before adjustment, {after:.2f} after weighting on the score"))
    return out, facts


# ------------------------------------------------------------------ helpers


def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    parts = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 2:
            v = s.astype(float)
            sd = v.std() or 1.0
            parts.append(((v - v.mean()) / sd).to_numpy()[:, None])
        else:
            parts.append(pd.get_dummies(s.astype(str), drop_first=True, dtype=float).to_numpy())
    return np.hstack(parts) if parts else np.zeros((len(df), 0))


def _propensity(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if X.shape[1] == 0:
        return np.full(len(y), y.mean()), 0.5
    m = LogisticRegression(max_iter=1000).fit(X, y)
    p = m.predict_proba(X)[:, 1]
    return p, float(roc_auc_score(y, p))


def _wmean(v: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(v, weights=w)) if w.sum() > 0 else float(v.mean())


def _wvar(v: np.ndarray, w: np.ndarray) -> float:
    m = _wmean(v, w)
    return float(np.average((v - m) ** 2, weights=w)) if w.sum() > 0 else float(v.var())


def _smd(s: pd.Series, t: pd.Series, weights: np.ndarray | None = None) -> float:
    """The standardised mean difference between arms, unweighted, or weighted (the balance after weighting on the score)."""
    tt = t.to_numpy()
    w = np.ones(len(s)) if weights is None else np.asarray(weights, dtype=float)
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 2:
        v = s.astype(float).to_numpy()
        a, b = v[tt], v[~tt]
        pooled = np.sqrt((_wvar(a, w[tt]) + _wvar(b, w[~tt])) / 2) or 1.0
        return float(abs(_wmean(a, w[tt]) - _wmean(b, w[~tt])) / pooled)
    d = pd.get_dummies(s.astype(str), dtype=float)
    worst = 0.0
    for col in d.columns:
        v = d[col].to_numpy()
        pa, pb = _wmean(v[tt], w[tt]), _wmean(v[~tt], w[~tt])
        pooled = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
        worst = max(worst, float(abs(pa - pb) / pooled) if pooled > 0 else 0.0)
    return worst
