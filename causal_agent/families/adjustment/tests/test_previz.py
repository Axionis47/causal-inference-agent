"""The adjustment family's pre-run figure on the real file: the overlap shares and the probe from the same cells."""

from __future__ import annotations

import pandas as pd

from causal_agent.families.adjustment.overlap import MAX_LEVELS
from causal_agent.families.adjustment.previz import overlap
from causal_agent.profile.datasets import ROOT

STUDENTS = pd.read_csv(ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv")


def test_overlap_shares_by_arm_and_the_probe_from_the_same_cells():
    f = overlap(STUDENTS, "test preparation course", "completed", ["lunch", "parental level of education"], floor=5, addresses=["claim:assignment.depends_on"])
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
    assert not overlap(STUDENTS, "nope", "x", ["lunch"]).made
    f = overlap(STUDENTS, "test preparation course", "completed", [])
    assert not f.made and "nothing the offer depended on" in f.why
    f = overlap(STUDENTS, "test preparation course", "finished", ["lunch"])
    assert not f.made and "one arm is empty" in f.why


def test_overlap_bins_a_numeric_column_and_flags_a_thin_cell():
    df = pd.DataFrame({"t": ["a"] * 30 + ["b"] * 30, "age": list(range(30)) + list(range(30, 60))})
    f = overlap(df, "t", "a", ["age"], floor=5)
    assert f.made and not f.probe.passed and "one arm missing" in f.probe.detail
    assert all(x.startswith("age = ") for x in f.spec.series[0].x) and len(f.spec.series[0].x) <= MAX_LEVELS
