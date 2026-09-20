"""The pre-viz functions on the real files: a figure with addresses and the probe number from the same computation."""

from __future__ import annotations

import pandas as pd

from causal_agent.memory.overlap import MAX_LEVELS
from causal_agent.profile.datasets import ROOT
from causal_agent.viz.previz import adjustment, diff_in_diff, discontinuity

STUDENTS = pd.read_csv(ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv")
SENATE = pd.read_csv(ROOT / "data/raw/senate-incumbency/senate.csv")


def test_overlap_shares_by_arm_and_the_probe_from_the_same_cells():
    f = adjustment.overlap(
        STUDENTS, "test preparation course", "completed", ["lunch", "parental level of education"], floor=5, addresses=["claim:assignment.depends_on"]
    )
    assert f.made and f.function == "adjustment.overlap" and f.spec.kind == "bars"
    treated, control = f.spec.series
    assert treated.x == control.x and treated.x[:2] == ["lunch = free/reduced", "lunch = standard"]
    assert abs(sum(treated.y[:2]) - 1) < 1e-9 and abs(sum(control.y[:2]) - 1) < 1e-9  # shares within an arm sum to one per column
    assert sum(treated.n[:2]) == int((STUDENTS["test preparation course"] == "completed").sum())
    assert f.probe.address == "probe:adjustment.overlap" and f.probe.passed and f.probe.value >= 5
    assert "probe:adjustment.overlap" in f.spec.draws_on and "claim:assignment.depends_on" in f.spec.draws_on
    assert "figure:overlap_lunch_parental_level_of_education.test_preparation_course_completed.0" in f.spec.addresses()
    text = f.spec.render()
    assert "[figure:overlap_lunch_parental_level_of_education]" in text and "lunch = standard" in text


def test_overlap_refuses_with_a_reason():
    assert not adjustment.overlap(STUDENTS, "nope", "x", ["lunch"]).made
    f = adjustment.overlap(STUDENTS, "test preparation course", "completed", [])
    assert not f.made and "nothing the offer depended on" in f.why
    f = adjustment.overlap(STUDENTS, "test preparation course", "finished", ["lunch"])
    assert not f.made and "one arm is empty" in f.why


def test_overlap_bins_a_numeric_column_and_flags_a_thin_cell():
    df = pd.DataFrame({"t": ["a"] * 30 + ["b"] * 30, "age": list(range(30)) + list(range(30, 60))})
    f = adjustment.overlap(df, "t", "a", ["age"], floor=5)
    assert f.made and not f.probe.passed and "one arm missing" in f.probe.detail
    assert all(x.startswith("age = ") for x in f.spec.series[0].x) and len(f.spec.series[0].x) <= MAX_LEVELS


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
