"""Synthetic eval fixtures with known ground truth.

Each generator implements its manifest recipe exactly (case ids in
representative_cases.yaml). Seeds are fixed so evals can assert
magnitudes. These are synthetic_eval_fixture cases: labeled synthetic
everywhere, never passed off as real Kaggle data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

GROUND_TRUTH = {
    "synthetic-did-panel": {"att": 2.0, "baseline_gap": 1.5},
    "synthetic-scholarship-rdd": {"jump": 8.0, "fuzzy_itt": 6.0, "cutoff": 50.0},
    "synthetic-iv-late": {"late": 2.0},
    "synthetic-mediation": {"indirect": 0.30, "direct": 0.30, "total": 0.60},
    "website-visitors-its-step": {"step_ratio": 1.25, "cut_date": "2019-04-01"},
}


def did_panel() -> pd.DataFrame:
    """60 units x 8 periods, treated jump +2.0 after period 4, ATT = 2.0."""
    rng = np.random.default_rng(42)
    units = np.arange(60)
    treated = units < 30
    alpha = np.where(treated, rng.normal(11.5, 2, 60), rng.normal(10, 2, 60))
    rows = []
    for u in units:
        for t in range(8):
            post = int(t >= 4)
            y = alpha[u] + 0.3 * t + 2.0 * int(treated[u]) * post + rng.normal(0, 1)
            rows.append(
                {
                    "unit_id": int(u),
                    "group": "treated" if treated[u] else "control",
                    "period": t,
                    "post": post,
                    "outcome": round(float(y), 4),
                }
            )
    return pd.DataFrame(rows)


def scholarship_rdd() -> pd.DataFrame:
    """Sharp and fuzzy RDD at score 50; jump 8.0; fuzzy take-up 0.15/0.90."""
    rng = np.random.default_rng(42)
    n = 2000
    score = np.round(rng.uniform(0, 100, n), 2)
    elig = (score >= 50).astype(int)
    d_sharp = elig
    y_sharp = 20 + 0.3 * score + 8.0 * d_sharp + rng.normal(0, 4, n)
    p_take = np.where(elig == 1, 0.90, 0.15)
    d_fuzzy = rng.binomial(1, p_take)
    y_fuzzy = 20 + 0.3 * score + 8.0 * d_fuzzy + rng.normal(0, 4, n)
    return pd.DataFrame(
        {
            "score": score,
            "eligible": elig,
            "scholarship_sharp": d_sharp,
            "outcome_sharp": np.round(y_sharp, 3),
            "scholarship_fuzzy": d_fuzzy,
            "outcome_fuzzy": np.round(y_fuzzy, 3),
        }
    )


def synthetic_iv() -> pd.DataFrame:
    """Z -> X -> Y with confounder U; no defiers; true LATE = ATE = 2.0."""
    rng = np.random.default_rng(47)
    n = 5000
    u = rng.normal(0, 1, n)
    expit = lambda v: 1.0 / (1.0 + np.exp(-v))  # noqa: E731
    p_always = expit(-2.2 + 0.8 * u)
    p_never = expit(-2.2 - 0.8 * u)
    draw = rng.uniform(0, 1, n)
    ctype = np.where(draw < p_always, "always", np.where(draw < p_always + p_never, "never", "complier"))
    z = rng.binomial(1, 0.5, n)
    x = np.where(ctype == "always", 1, np.where(ctype == "never", 0, z))
    c1 = rng.normal(0, 1, n)
    c2 = rng.binomial(1, 0.4, n)
    y = 2.0 * x + 1.5 * u + 0.8 * c1 + 0.5 * c2 + rng.normal(0, 1, n)
    return pd.DataFrame(
        {"z": z, "x": x, "y": np.round(y, 4), "c1": np.round(c1, 4), "c2": c2}
    )


def mediation() -> pd.DataFrame:
    """Randomized T; M mediates half the total effect (0.30 of 0.60)."""
    rng = np.random.default_rng(42)
    n = 2000
    c = rng.normal(0, 1, n)
    t = rng.binomial(1, 0.5, n)
    m = 0.6 * t + 0.4 * c + rng.normal(0, 1, n)
    y = 0.5 * m + 0.3 * t + 0.5 * c + rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "exercise_program": t,
            "weight_change": np.round(m, 4),
            "disease_risk": np.round(y, 4),
            "baseline_health": np.round(c, 4),
        }
    )


def website_visitors_step(base_csv: Path) -> pd.DataFrame:
    """+25% multiplicative step at 2019-04-01 injected into the real series."""
    rng = np.random.default_rng(42)
    df = pd.read_csv(base_csv, thousands=",")
    df["Date"] = pd.to_datetime(df["Date"])
    window = df[(df["Date"] >= "2018-04-01") & (df["Date"] < "2020-04-01")].copy()
    step = np.where(window["Date"] >= "2019-04-01", 1.25, 1.0)
    noise = rng.normal(1.0, 0.02, len(window))
    window["visits"] = np.round(window["Unique.Visits"] * step * noise).astype(int)
    return window[["Date", "Day", "visits"]].reset_index(drop=True)


GENERATORS = {
    "synthetic-did-panel": did_panel,
    "synthetic-scholarship-rdd": scholarship_rdd,
    "synthetic-iv-late": synthetic_iv,
    "synthetic-mediation": mediation,
}
