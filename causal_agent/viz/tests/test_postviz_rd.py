"""The discontinuity lane's figures: the jump with its fits, the density either side, the covariates' jumps, the bandwidth curve, the placebo cutoffs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_agent.viz.graph import check_spec
from causal_agent.viz.postviz import discontinuity as R


def canon(n=2000, jump=2.0, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 1 + x + jump * (x >= 0) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x": x, "y": y, "side": (x >= 0).astype(int)})


def test_rd_plot_fits_each_side_and_reads_the_jump():
    c = canon()
    b = pd.DataFrame({"rdplot_mean_x": [-0.75, -0.25, 0.25, 0.75], "rdplot_mean_y": [0.25, 0.75, 3.25, 3.75], "rdplot_N": [500, 500, 500, 500]})
    f = R.rd_plot(b, c, h=0.5, p=1, contrast="above_vs_below", score_name="margin")
    assert f.kind == "points" and f.id == "rd_plot_above_vs_below" and [s.name for s in f.series] == ["binned means", "fit, control side", "fit, treated side"]
    assert f.series[0].n == [500, 500, 500, 500] and len(f.series[1].x) == 40 and f.series[1].x[-1] == 0.0 and f.series[2].x[0] == 0.0
    jump = f.series[2].y[0] - f.series[1].y[-1]
    assert 1.7 < jump < 2.3 and "apart" in f.note and f.marks[0].at == 0.0
    assert set(f.draws_on) == {"design.bandwidth", "estimate:above_vs_below.value"} and check_spec(f, set(f.draws_on)) == []
    assert R.rd_plot(None, c, h=0.5, p=1, contrast="c").series[0].name == "fit, control side"
    assert R.rd_plot(b, pd.DataFrame(), h=0.5, p=1, contrast="c") is None


def test_density_either_side():
    c = canon()
    f = R.density_test(c["x"], {"computable": True, "p": 0.42, "hat_left": 0.5, "hat_right": 0.51}, "above_vs_below", bins=20)
    assert f.kind == "density" and [s.name for s in f.series] == ["below the cutoff", "at or above the cutoff"]
    assert all(x < 0 for x in f.series[0].x) and all(x >= 0 for x in f.series[1].x) and sum(f.series[0].y) + sum(f.series[1].y) == 2000
    assert "p = 0.42" in f.note and "no sign of bunching" in f.note and f.draws_on == ["check:above_vs_below.density"]
    g = R.density_test(c["x"], {"computable": False, "reason": "sampled by side"}, "c")
    assert "could not be read" in g.note


def test_continuity_bandwidths_and_placebo_cutoffs():
    f = R.covariate_continuity({"age": {"jump": 0.1, "ci_low": -0.2, "ci_high": 0.4, "p": 0.5}, "education": {"jump": 0.9, "ci_low": 0.3, "ci_high": 1.5, "p": 0.01}}, "c", {"education": "years of schooling"})
    assert f.kind == "interval" and f.series[0].x == ["age", "years of schooling"] and f.series[0].lo == [-0.2, 0.3] and "years of schooling differ" in f.note
    pts = [{"label": "h_cer = 0.3", "at": 0.3, "value": 2.1, "lo": 1.5, "hi": 2.7, "n_l": 300, "n_r": 300, "informative": True},
           {"label": "2h_mse = 1.0", "at": 1.0, "value": 1.9, "lo": 1.6, "hi": 2.2, "n_l": 1000, "n_r": 1000, "informative": True},
           {"label": "h_mse = 0.5", "at": 0.5, "value": 2.0, "lo": 1.6, "hi": 2.4, "n_l": 500, "n_r": 500, "informative": True},
           {"label": "none", "informative": False, "note": "no coverage-error bandwidth", "at": None}]
    g = R.bandwidth_curve(pts, 0.5, "c")
    assert g.series[0].x == [0.3, 0.5, 1.0] and g.series[0].y == [2.1, 2.0, 1.9] and g.marks[0].at == 0.5 and g.series[0].n == [600, 1000, 2000]
    assert set(g.draws_on) == {"placebo:c.bandwidth_grid.detail", "design.bandwidth"}
    pc = R.placebo_cutoffs([{"label": "control side at -0.5", "at": -0.5, "value": 0.05, "lo": -0.2, "hi": 0.3}, {"label": "treated side at 0.5", "at": 0.5, "value": -0.1, "lo": -0.4, "hi": 0.2}],
                           {"value": 2.0, "ci_low": 1.6, "ci_high": 2.4}, "c")
    assert pc.series[0].x == ["control side at -0.5", "the cutoff", "treated side at 0.5"] and pc.series[0].y[1] == 2.0
    assert R.bandwidth_curve([], 0.5, "c") is None and R.placebo_cutoffs([], {}, "c") is None and R.covariate_continuity({}, "c") is None
