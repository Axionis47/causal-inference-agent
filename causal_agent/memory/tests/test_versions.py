"""The versions shelf and the switch: a version is written beside its parent under the memory home, listed and loaded by number;
putting one in play re-reads the file's facts and columns, keeps a dropped column's fields out of play, and moves the version."""

from __future__ import annotations

import pandas as pd
import pytest

from causal_agent.common import config
from causal_agent.memory import store
from causal_agent.memory import versions as VS
from causal_agent.memory.tests.test_memory import ROOT
from causal_agent.profile.profiler import profile


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path


def _version(v: int, parent: int = 0) -> VS.Version:
    return VS.Version(
        v=v,
        dataset="students",
        parent=parent,
        ask="drop the writing score",
        summary="one column fewer",
        rows=1000,
        columns=["a"],
        dropped=["writing score"],
        memory_version=3,
        made_at=f"2026-10-06T0{v}:00:00+00:00",
    )


def test_a_version_is_written_beside_its_parent_listed_and_loaded(root):
    assert VS.next_version("students") == 1 and VS.list_versions("students") == [] and VS.table_path("students", 0) is None
    d = VS.save(_version(1))
    assert d == root / "data" / "memory" / "students" / "versions" / "1" and (d / VS.FILE).is_file()
    VS.save(_version(2, parent=1))
    assert VS.next_version("students") == 3 and [x.v for x in VS.list_versions("students")] == [1, 2]
    v = VS.load("students", 2)
    assert v.parent == 1 and v.address == "version:2" and VS.table_path("students", 2) == d.parent / "2" / VS.TABLE
    assert v.render() == "[version:2] table version 2 from 1: one column fewer (1000 rows; dropped writing score)"


def test_switching_the_version_in_play_rereads_the_file_and_moves_the_memory(root, tmp_path):
    m = store.migrate("students3", root=ROOT, write=False)
    df = pd.read_csv(ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv")
    df["high reading"] = (df["reading score"] >= 70).astype(int)
    new = tmp_path / "table.csv"
    df.drop(columns=["writing score"]).to_csv(new, index=False)
    before, v = set(m.columns), m.version
    when = m.field("col:writing_score.when")
    assert when is not None and "writing_score" in before
    added, dropped = m.switch_version(1, str(new), profile(new))
    assert (added, dropped) == (["high_reading"], ["writing_score"])
    assert m.data_version == 1 and m.csv == str(new) and m.version == v + 1
    assert "writing_score" not in m.columns and m.columns["high_reading"].name == "high reading"
    assert m.field("col:writing_score.when") is when, "a dropped column's fields stay in the map, out of play"
    assert m.facts["columns"] == len(df.columns) - 1
    assert "high reading" in m.render() and "col:writing_score" not in m.render()
    table = m.to_claims()
    assert "col:high_reading" in table.claims and "col:writing_score" not in table.claims
