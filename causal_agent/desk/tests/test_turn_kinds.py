"""The kind of a turn is read first and gated by code: only update carries writes, explore and ask carry what they are about,
a note is always update. A reading that breaks that is re-prompted with the reason and writes nothing."""

from __future__ import annotations

from pathlib import Path

from causal_agent.common.llm import set_llm
from causal_agent.desk import reader as RD
from causal_agent.desk.contracts import FieldUpdate, Reading
from causal_agent.memory import store

REPO = Path(__file__).resolve().parents[3]


def test_the_kind_gate_names_each_break():
    ask_with_writes = Reading(kind="ask", question="why?", updates=[FieldUpdate(address="col:lunch.when", value="before", said="x")])
    assert RD.kind_errors(ask_with_writes, from_note=False) == [
        "kind is ask but the reading carries writes; a message that says anything about the data is update"
    ]
    assert "explore is empty" in RD.kind_errors(Reading(kind="explore"), from_note=False)[0]
    assert "question is empty" in RD.kind_errors(Reading(kind="ask"), from_note=False)[0]
    assert "a note is always read as update" in RD.kind_errors(Reading(kind="run"), from_note=True)[0]
    assert RD.kind_errors(Reading(kind="update", question="and why?", explore="and a plot"), from_note=False) == []
    assert RD.kind_errors(Reading(kind="run"), from_note=False) == []


class _Fixed:
    """A model that answers every reading the same way."""

    def __init__(self, out: Reading):
        self.out = out

    def with_structured_output(self, schema, include_raw=False):
        out = self.out

        class R:
            def invoke(self_, messages):
                return {"raw": None, "parsed": out, "parsing_error": None}

        return R()


def test_a_note_read_as_anything_but_update_is_refused_and_writes_nothing():
    memory = store.migrate("students", root=REPO, write=False)
    before = memory.version
    set_llm(_Fixed(Reading(kind="ask", question="what is this?")))
    out, _thoughts, rejected = RD.read_words(memory, "doc:students", "a note", RD.DESCRIPTION, [], 0)
    assert out.kind == "ask" and rejected == ["kind is ask but a note is always read as update"]
    assert memory.version == before
    set_llm(None)
