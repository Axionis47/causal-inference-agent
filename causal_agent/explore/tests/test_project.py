"""Before the freeze the table sent has one of the pair taken out, by code: the outcome unless the ask names the outcome and not
the treatment; an ask naming both is refused; after the freeze nothing is dropped."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from causal_agent.explore import project as PJ

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"


def test_which_column_is_dropped_is_decided_by_the_ask():
    o, t = "math score", "test preparation course"
    assert PJ.choose_drop("counts by lunch", o, t, frozen=False) == "outcome"
    assert PJ.choose_drop("how does math_score look by lunch", o, t, frozen=False) == "treatment"  # the key spells the outcome too
    assert PJ.choose_drop("who completed the test preparation course", o, t, frozen=False) == "outcome"
    assert PJ.choose_drop("math score by test preparation course", o, t, frozen=False) == "both"
    assert PJ.choose_drop("math score by test preparation course", o, t, frozen=True) is None
    assert PJ.choose_drop("anything", None, t, frozen=False) is None and PJ.choose_drop("anything", o, None, frozen=False) is None


def test_the_projection_keeps_every_row_and_drops_one_column(tmp_path):
    path = PJ.project(CSV, tmp_path / "art", drop="math score")
    assert path == tmp_path / "art" / "input" / "table.csv"
    df, full = pd.read_csv(path), pd.read_csv(CSV)
    assert len(df) == len(full) == 1000 and "math score" not in df.columns and set(df.columns) == set(full.columns) - {"math score"}
