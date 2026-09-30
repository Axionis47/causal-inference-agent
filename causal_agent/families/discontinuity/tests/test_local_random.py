"""Local randomisation on a discrete score: the window by balance, the estimate in a window with its inverted interval, the
sensitivity across windows and the Rosenbaum bounds. The library's own interval does not run in this port; the adapter inverts
the test itself, and these tests hold that to the truth planted in the toy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_agent.families.discontinuity.lane import adapter

REPS = 200


def discrete_canon(n: int = 1500, effect: float = 1.5, seed: int = 3, fuzzy: bool = False) -> pd.DataFrame:
    """Twelve integer scores, the line between 6 and 7, recentred so the treated side is x >= 0; two covariates fixed before."""
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 13, n).astype(float) - 7.0
    age = 30 + 0.3 * x + rng.normal(0, 4, n)
    income = 50 + 0.5 * x + rng.normal(0, 8, n)
    df = pd.DataFrame({"x": x, "cov_age": age, "cov_income": income})
    if fuzzy:
        df["t"] = ((x >= 0) & (rng.random(n) < 0.7)).astype(float)
        df["y"] = 2 + 0.2 * x + effect * df["t"] + rng.normal(0, 1, n)
    else:
        df["y"] = 2 + 0.2 * x + effect * (x >= 0) + rng.normal(0, 1, n)
    return df


def test_the_window_by_balance_lists_nested_windows_and_recommends_one():
    canon = discrete_canon()
    w = adapter.local_random_windows(canon, ["cov_age", "cov_income"], reps=REPS, seed=7)
    assert "error" not in w and w["w_left"] is not None and w["w_right"] is not None and w["w_left"] < 0 <= w["w_right"]
    assert len(w["windows"]) >= 3 and all(r["n_left"] > 0 and r["n_right"] > 0 for r in w["windows"])
    assert all(r["p_min"] is not None for r in w["windows"]) and all(0 <= r["binomial_p"] <= 1 for r in w["windows"])
    # without covariates the library recommends no window; the caller falls back
    w0 = adapter.local_random_windows(canon, [], reps=REPS, seed=7)
    assert w0["w_left"] is None and w0["windows"] and w0["windows"][0]["p_min"] is None


def test_the_estimate_in_a_window_recovers_the_planted_jump_with_an_inverted_interval():
    canon = discrete_canon(effect=1.5)
    # scores 6 and 7 only; the difference in means carries the jump plus one step of the slope (0.2), as local randomisation does
    f = adapter.fit_local_random(canon, -1.0, 0.0, reps=REPS, seed=7)
    assert f.error is None and f.model.endswith("difference in means") and f.vce == "randomisation"
    assert f.value == pytest.approx(1.7, abs=0.3) and f.ci_low < 1.7 < f.ci_high and f.ci_high - f.ci_low < 1.0
    assert f.p < 0.01 and f.n_h_left > 50 and f.n_h_right > 50 and f.h_left == 1.0 and f.h_right == 0.0
    # no effect: the interval covers the slope step and the p is large against zero only when the slope is absent
    flat = discrete_canon(effect=0.0, seed=11)
    flat["y"] = flat["y"] - 0.2 * flat["x"]
    g = adapter.fit_local_random(flat, -1.0, 0.0, reps=REPS, seed=7)
    assert g.error is None and g.ci_low < 0 < g.ci_high and g.p > 0.05


def test_a_fuzzy_design_uses_the_anderson_rubin_statistic_and_the_wald_ratio():
    canon = discrete_canon(effect=2.0, fuzzy=True)
    f = adapter.fit_local_random(canon, -1.0, 0.0, fuzzy=True, reps=REPS, seed=7)
    assert f.error is None and f.model.endswith("Anderson-Rubin")
    wald = (0.2 + 2.0 * 0.7) / 0.7  # the outcome's jump over take-up's jump, slope step included
    assert f.value == pytest.approx(wald, abs=0.6) and f.ci_low < wald < f.ci_high


def test_sensitivity_across_windows_and_the_rosenbaum_bounds():
    canon = discrete_canon(effect=1.5)
    rows = adapter.window_sensitivity(canon, [(-1.0, 0.0), (-2.0, 1.0), (-3.0, 2.0)], fuzzy=False, reps=REPS, seed=7)
    assert [r["w_right"] for r in rows] == [0.0, 1.0, 2.0] and all(r["error"] is None for r in rows)
    assert all(r["ci_low"] < 1.5 + 0.6 and r["ci_high"] > 1.5 - 0.6 for r in rows)  # the slope widens the window's estimate, the truth stays inside
    b = adapter.rosenbaum_bounds(canon, 1.0, [0.1, 0.5, 1.0], reps=REPS, seed=7)
    assert "error" not in b and b["gamma"] == pytest.approx([0.1, 0.5, 1.0]) and len(b["lower"]) == 3 == len(b["upper"])
    assert all(lo <= hi for lo, hi in zip(b["lower"], b["upper"], strict=True))


def test_a_window_that_holds_no_rows_is_an_error_not_an_exception():
    canon = discrete_canon()
    assert adapter.fit_local_random(canon, 0.2, 0.8, reps=REPS, seed=7).error is not None  # no integer score inside
    assert adapter.fit_local_random(canon, 0.0, 0.0, reps=REPS, seed=7).error is not None  # the library refuses wl == wr
