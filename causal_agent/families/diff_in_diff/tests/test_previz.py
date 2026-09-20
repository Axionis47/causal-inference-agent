"""The diff-in-diff family's pre-run figure: the two groups' paths with the change marked, and the pre-period count."""

from __future__ import annotations

import pandas as pd

from causal_agent.families.diff_in_diff.previz import by_group_over_time


def test_by_group_over_time_marks_the_change_and_counts_pre_periods():
    rows = []
    for unit in range(6):
        for year in range(2000, 2008):
            rows.append({"unit": unit, "year": year, "treated": "yes" if unit < 3 else "no", "y": year - 2000 + (3 if unit < 3 and year >= 2004 else 0)})
    df = pd.DataFrame(rows)
    f = by_group_over_time(df, "y", "year", "treated", "yes", 2004, floor=2)
    assert f.made and f.spec.kind == "lines" and [s.name for s in f.spec.series] == ["got the change", "did not"]
    assert f.spec.series[0].x == [float(y) for y in range(2000, 2008)]
    assert f.spec.series[0].y[4] - f.spec.series[1].y[4] == 3 and f.spec.series[0].y[3] == f.spec.series[1].y[3]
    assert f.spec.marks[0].at == 2004.0 and f.probe.name == "pre_periods" and f.probe.value == 4 and f.probe.passed
    g = by_group_over_time(df, "y", "year", "treated", "yes", None)
    assert g.made and g.probe.passed is None and g.spec.marks == []
    assert not by_group_over_time(df, "y", "year", "treated", "maybe", 2004).made
