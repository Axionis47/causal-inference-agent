"""The diff-in-diff lane's figures: the paths with the treated group's path without the change, the placebo spread."""

from __future__ import annotations

import pandas as pd

from causal_agent.families.diff_in_diff import postviz as D
from causal_agent.viz.graph import check_spec


def toy_panel() -> pd.DataFrame:
    rows = []
    for unit in range(6):
        treated = 1 if unit < 3 else 0
        for t in range(4):
            post = 1 if t >= 2 else 0
            rows.append({"unit": str(unit), "time": t, "treated": treated, "post": post, "y": 10 + t + 5 * treated + 3 * treated * post})
    return pd.DataFrame(rows)


def test_paths_with_the_counterfactual():
    f = D.paths_with_counterfactual(toy_panel(), "1_vs_0", estimate=3.0)
    assert f.kind == "lines" and f.id == "paths_1_vs_0" and [s.name for s in f.series] == ["got the change", "did not", "the treated group without the change"]
    got, not_, cf = f.series
    assert got.x == [0.0, 1.0, 2.0, 3.0] and got.y == [15, 16, 20, 21] and not_.y == [10, 11, 12, 13] and got.n == [3, 3, 3, 3]
    # the treated group's own pre level (15.5) plus the comparison group's movement from its pre level (10.5)
    assert cf.y[:2] == [None, None] and cf.y[2] == 17.0 and cf.y[3] == 18.0
    assert f.marks[0].kind == "vline" and f.marks[0].at == 2.0 and "3" in f.note
    assert set(f.draws_on) == {"design.periods", "estimate:1_vs_0.value"} and check_spec(f, set(f.draws_on)) == []
    assert D.paths_with_counterfactual(pd.DataFrame(), "c") is None


def test_placebo_distribution():
    draws = [0.1 * i - 1.0 for i in range(21)]
    f = D.placebo_distribution(draws, observed=3.0, p=0.0, contrast="1_vs_0", bins=10)
    assert f.kind == "density" and f.id == "placebo_1_vs_0" and len(f.series[0].x) == 10 and sum(f.series[0].y) == 21
    assert f.marks[0].at == 3.0 and "21 reassignments" in f.note and "0.00" in f.note
    assert set(f.draws_on) == {"placebo:1_vs_0.placebo_group.p_value", "estimate:1_vs_0.value"}
    assert D.placebo_distribution([1.0, 2.0], 0.5, 0.5, "c") is None
