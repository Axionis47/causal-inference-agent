"""Executable pre-freeze requests always lose the outcome, even when assignment is available through other columns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from causal_agent.explore import project as PJ

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"


def test_every_executable_pre_freeze_request_withholds_the_outcome():
    o, t = "math score", "test preparation course"
    assert PJ.choose_drop("counts by lunch", o, t, frozen=False) == "outcome"
    assert PJ.choose_drop("how does math_score look by lunch", o, t, frozen=False) == "outcome"
    assert PJ.choose_drop("who completed the test preparation course", o, t, frozen=False) == "outcome"
    assert PJ.choose_drop("math score by test preparation course", o, t, frozen=False) == "both"
    assert PJ.choose_drop("math score by test preparation course", o, t, frozen=True) is None
    assert PJ.choose_drop("anything", None, t, frozen=False) is None
    assert PJ.choose_drop("anything", o, None, frozen=False) == "outcome"


@pytest.mark.parametrize(
    "text,column", [("numpy pyplot yearly", "y"), ("describe age by group", "y"), ("yesterday", "y"), ("xylophone", "y"), ("histogram currently", "t")]
)
def test_short_columns_match_complete_tokens_only(text, column):
    assert not PJ.mentions(text, column)
    assert PJ.choose_drop(text, column, "group", frozen=False) == "outcome"


@pytest.mark.parametrize("text", ["describe y", "y by group", "mean_y_by_group", "math_score", "math score"])
def test_column_words_and_keys_match(text):
    assert PJ.mentions(text, "math score" if text.startswith("math") else "y")


@pytest.mark.parametrize("assignment", ["forcing_score", "instrument", "cohort", "group", "adoption_date"])
@pytest.mark.parametrize("ask", ["outcome by {assignment}", "show all columns", "describe outcome"])
def test_assignment_proxies_never_share_the_projected_table_with_outcome(tmp_path, assignment, ask):
    csv = tmp_path / "data.csv"
    original = pd.DataFrame({"outcome": [1.0, 2.0], "treatment": [0, 1], assignment: [0, 1]})
    original.to_csv(csv, index=False)
    drop = PJ.choose_drop(ask.format(assignment=assignment), "outcome", "treatment", frozen=False)
    assert drop == "outcome"
    sent = pd.read_csv(PJ.project(csv, tmp_path / "artifact", drop="outcome" if drop == "outcome" else None))
    assert "outcome" not in sent.columns
    pd.testing.assert_frame_equal(sent, original.drop(columns="outcome"))


def test_frozen_projection_retains_the_outcome(tmp_path):
    assert PJ.choose_drop("math score by lunch", "math score", "test preparation course", frozen=True) is None
    pd.testing.assert_frame_equal(pd.read_csv(PJ.project(CSV, tmp_path / "artifact", drop=None)), pd.read_csv(CSV))


def test_the_projection_keeps_every_row_and_drops_one_column(tmp_path):
    path = PJ.project(CSV, tmp_path / "art", drop="math score")
    assert path == tmp_path / "art" / "input" / "table.csv"
    df, full = pd.read_csv(path), pd.read_csv(CSV)
    assert len(df) == len(full) == 1000 and "math score" not in df.columns and set(df.columns) == set(full.columns) - {"math score"}
