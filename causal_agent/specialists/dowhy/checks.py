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


def run_checks(table: pd.DataFrame, adjustment_set: list[str], contrast_key: str, cfg: dict[str, Any]) -> list[CheckResult]:
    out: list[CheckResult] = []
    t = table[TREATED] == 1
    n_t, n_c = int(t.sum()), int((~t).sum())
    thr = float(cfg["arms"]["min_per_arm"]["hard"])
    out.append(CheckResult(contrast=contrast_key, name="arms", level="hard" if min(n_t, n_c) < thr else "pass",
                           value=float(min(n_t, n_c)), threshold=thr, detail=f"{n_t} treated, {n_c} control"))
    if not adjustment_set:
        out.append(CheckResult(contrast=contrast_key, name="overlap", level="pass", detail="nothing to adjust for; overlap holds by construction"))
        return out

    X = _design_matrix(table[adjustment_set])
    scores, auc = _propensity(X, t.astype(int).to_numpy())
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
        level = "hard" if smd >= b["hard"] else "soft" if smd >= b["soft"] else "pass"
        out.append(CheckResult(contrast=contrast_key, name=f"balance.{col}", level=level, value=round(smd, 3), threshold=b["soft"],
                               detail=f"standardised mean difference {smd:.2f} between arms before adjustment"))
    return out


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


def _smd(s: pd.Series, t: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 2:
        a, b = s[t].astype(float), s[~t].astype(float)
        pooled = np.sqrt((a.var() + b.var()) / 2) or 1.0
        return float(abs(a.mean() - b.mean()) / pooled)
    d = pd.get_dummies(s.astype(str), dtype=float)
    worst = 0.0
    for col in d.columns:
        a, b = d[col][t], d[col][~t]
        pa, pb = a.mean(), b.mean()
        pooled = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
        worst = max(worst, float(abs(pa - pb) / pooled) if pooled > 0 else 0.0)
    return worst
