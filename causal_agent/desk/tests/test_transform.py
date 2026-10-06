"""A change to the table is a new version in play: the data agent writes it beside the one before, the memory switches to it on
the person's word with the journal recording the step, the interview goes on over the new table, the run reads it, and after a
run a change goes back to the ready moment on the new version; nothing reruns on its own."""

from __future__ import annotations

import json
from pathlib import Path

from causal_agent.desk.contracts import AfterReply
from causal_agent.desk.tests.fakes import QUESTION, DeskFake
from causal_agent.desk.tests.test_journey import HELD, Desk, _held  # noqa: F401  (the autouse fixture that holds the memory in process)
from causal_agent.memory import versions as VS


def test_a_change_makes_a_version_the_memory_switches_to_and_the_run_reads(tmp_path):
    fake = DeskFake(
        after=[AfterReply(kind="transform", text="I will make a new table.", transform="drop the writing score column"), AfterReply(kind="done", text="Bye.")]
    )
    d = Desk(fake)
    d.say(QUESTION)
    reads = fake.calls.count("Reading")
    p = d.say("add a column high_reading that is 1 when reading score is 70 or more")
    m = HELD["students"]
    assert p["kind"] == "chat" and p["text"].startswith("Table version 1 is now in play: Added the high reading flag") and "[version:1]" in p["text"]
    assert fake.calls.count("Reading") == reads, "the decision model decided transform; no Reader"
    folder = tmp_path / "data" / "memory" / "students" / "versions" / "1"
    assert m.data_version == 1 and m.csv == str(folder / "table.csv") and "high_reading" in m.columns and m.columns["high_reading"].name == "high reading"
    v = VS.load("students", 1)
    assert v.parent == 0 and v.added == ["high reading"] and v.rows == 1000
    step = d.journal.last("version")
    assert step is not None and step.by == "person" and step.note == "version 1: " + v.summary and "version:0" in step.read
    assert "math score" not in (folder / "code.py").read_text(), "the outcome was held back from the agent"
    d.to_ready()
    assert d.values["preview"].data_version == 1 and d.values["preview"].csv == m.csv
    d.say("yes, run it")
    h = json.loads((Path(d.values["runs"][0].design_dir) / "handoff.json").read_text())
    assert h["data_version"] == 1 and h["csv"] == m.csv
    # after the run: a change makes version 2 from 1 and goes back to the ready moment on it
    p = d.say("drop the writing score column")
    assert m.data_version == 2 and VS.load("students", 2).parent == 1 and d.values["phase"] == "before"
    assert p["kind"] == "ready" and "Table version 2 is now in play" in p["text"]
    assert [s.kind for s in d.journal.steps() if s.kind == "version"] == ["version", "version"]
