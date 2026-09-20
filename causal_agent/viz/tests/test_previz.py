"""The pre-viz functions on the real files: a figure with addresses and the probe number from the same computation."""

from __future__ import annotations

import pandas as pd

from causal_agent.profile.datasets import ROOT
from causal_agent.viz.previz import diff_in_diff, discontinuity

SENATE = pd.read_csv(ROOT / "data/raw/senate-incumbency/senate.csv")


def test_by_group_over_time_marks_the_change_and_counts_pre_periods():
    rows = []
    for unit in range(6):
        for year in range(2000, 2008):
            rows.append({"unit": unit, "year": year, "treated": "yes" if unit < 3 else "no", "y": year - 2000 + (3 if unit < 3 and year >= 2004 else 0)})
    df = pd.DataFrame(rows)
    f = diff_in_diff.by_group_over_time(df, "y", "year", "treated", "yes", 2004, floor=2)
    assert f.made and f.spec.kind == "lines" and [s.name for s in f.spec.series] == ["got the change", "did not"]
    assert f.spec.series[0].x == [float(y) for y in range(2000, 2008)]
    assert f.spec.series[0].y[4] - f.spec.series[1].y[4] == 3 and f.spec.series[0].y[3] == f.spec.series[1].y[3]
    assert f.spec.marks[0].at == 2004.0 and f.probe.name == "pre_periods" and f.probe.value == 4 and f.probe.passed
    g = diff_in_diff.by_group_over_time(df, "y", "year", "treated", "yes", None)
    assert g.made and g.probe.passed is None and g.spec.marks == []
    assert not diff_in_diff.by_group_over_time(df, "y", "year", "treated", "maybe", 2004).made


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
