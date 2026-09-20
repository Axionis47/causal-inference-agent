"""The pre-viz functions on the real files: a figure with addresses and the probe number from the same computation."""

from __future__ import annotations

import pandas as pd

from causal_agent.profile.datasets import ROOT
from causal_agent.viz.previz import discontinuity

SENATE = pd.read_csv(ROOT / "data/raw/senate-incumbency/senate.csv")


def test_density_and_outcome_by_bin_around_the_cutoff():
    d = discontinuity.density(SENATE, "margin", 0.0, bins=10, window=50)
    assert d.made and d.spec.kind == "density" and d.spec.marks[0].at == 0.0
    xs = d.spec.series[0].x
    assert all(-50 <= x <= 50 for x in xs) and sum(1 for x in xs if x < 0) == 5 and sum(1 for x in xs if x > 0) == 5
    assert d.probe.name == "rows_by_side" and d.probe.passed
    below, above = discontinuity.rows_by_side(pd.to_numeric(SENATE["margin"], errors="coerce").dropna(), 0.0)
    assert d.probe.value == min(below, above)
    o = discontinuity.outcome_by_bin(SENATE, "margin", 0.0, "vote", bins=10, window=50)
    assert o.made and o.spec.kind == "points" and len(o.spec.series[0].x) == 10 and all(n > 0 for n in o.spec.series[0].n)
    assert o.spec.series[0].y[-1] > o.spec.series[0].y[0]  # a wider win margin sits with a larger vote share
    assert not discontinuity.density(SENATE, "nope", 0.0).made
