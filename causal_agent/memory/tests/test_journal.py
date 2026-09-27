"""The analysis journal: numbered, append-only, tolerant of a torn last line, and creating nothing until a step is written."""

from __future__ import annotations

from causal_agent.memory import journal as J


def test_steps_are_numbered_in_order_and_round_trip(tmp_path):
    j = J.open_journal("d", "a1", tmp_path)
    assert not J.home("d", "a1", tmp_path).exists()  # opening creates nothing
    s1 = j.append("question", by="model", memory_version=3, read=["user:turn:1", "col:math_score"], note="effect: math score against course")
    s2 = j.append("claim", by="person", memory_version=4, read=["user:turn:2"], note="claim:assignment.kind", at="2026-09-27T00:00:00+00:00")
    s3 = j.append("design", by="code", memory_version=4, design=1, left=[tmp_path / "data/memory/d/designs/1", "/elsewhere/run-x"], note="design 1")
    assert [s.n for s in (s1, s2, s3)] == [1, 2, 3] and s1.address == "step:1"
    back = j.steps()
    assert [s.model_dump() for s in back] == [s.model_dump() for s in (s1, s2, s3)]
    assert back[1].at == "2026-09-27T00:00:00+00:00" and back[0].at.endswith("+00:00")
    assert back[2].left == ["designs/1", "/elsewhere/run-x"]  # inside the memory dir: relative; outside: as given
    assert j.last().n == 3 and j.last("claim").n == 2 and j.last("run") is None
    assert j.resolve("step:2").kind == "claim" and j.resolve("2").kind == "claim" and j.resolve("step:9") is None and j.resolve("x") is None
    assert j.addresses() == {"step:1", "step:2", "step:3"}
    assert "design 1" in s3.line() and s3.line().startswith("design by code · v4 · design 1")


def test_a_torn_last_line_is_skipped_on_read_and_appended_past(tmp_path):
    j = J.open_journal("d", "a1", tmp_path)
    j.append("question", by="model", memory_version=1)
    j.append("claim", by="person", memory_version=2)
    with j.path.open("a") as f:
        f.write('{"n": 3, "kind": "cla')  # the process died mid-write
    assert [s.n for s in j.steps()] == [1, 2]
    s = j.append("claim", by="person", memory_version=3)
    assert s.n == 3 and [x.n for x in j.steps()] == [1, 2, 3]
    lines = j.path.read_text().splitlines()
    assert lines[-1].startswith('{"n":3,') and len(lines) == 4  # the torn line stays as its own line; nothing is glued to it


def test_analysis_ids_count_only_the_analyses(tmp_path):
    assert J.analysis_ids("d", tmp_path) == [] and J.next_analysis_id("d", tmp_path) == "a1"
    for a in ("a1", "a3", "a10", "notes"):
        (tmp_path / "data/memory/d/analyses" / a).mkdir(parents=True)
    assert J.analysis_ids("d", tmp_path) == ["a1", "a3", "a10"] and J.next_analysis_id("d", tmp_path) == "a11"


def test_the_note_is_one_short_line(tmp_path):
    j = J.open_journal("d", "a1", tmp_path)
    s = j.append("answer", by="model", memory_version=1, note="  " + "x" * 500 + "  ")
    assert len(s.note) == 300 and j.steps()[0].note == s.note
