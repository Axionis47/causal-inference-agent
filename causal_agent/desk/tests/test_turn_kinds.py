"""The kind of a turn is read first and gated by code: only update carries writes, explore and ask carry what they are about,
a note is always update. A reading that breaks that is re-prompted with the reason and writes nothing."""

from __future__ import annotations

from causal_agent.common.llm import set_llm
from causal_agent.desk import reader as RD
from causal_agent.desk.contracts import FieldUpdate, Reading
from causal_agent.desk.tests.fakes import QUESTION, DeskFake
from causal_agent.desk.tests.test_journey import HELD, Desk, _held  # noqa: F401  (the autouse fixture that holds the memory in process)
from causal_agent.memory import store


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
    memory = store.migrate("students", write=False)
    before = memory.version
    set_llm(_Fixed(Reading(kind="ask", question="what is this?")))
    out, _thoughts, rejected = RD.read_words(memory, "doc:students", "a note", RD.DESCRIPTION, [], 0)
    assert out.kind == "ask" and rejected == ["kind is ask but a note is always read as update"]
    assert memory.version == before
    set_llm(None)


def test_a_reading_with_the_wrong_kind_is_re_prompted_three_times_and_the_memory_is_untouched():
    def stubborn(msg, addrs, human):
        if msg.startswith("why"):
            return Reading(kind="ask", question=msg, updates=[FieldUpdate(address="claim:unobserved.exists", value="false", said=msg)])
        return None

    fake = DeskFake(infer=stubborn)
    d = Desk(fake)
    d.say(QUESTION)
    m = HELD["students"]
    version = m.version
    reads_before = len(fake.reads())
    d.say("why do you ask about hidden factors, I wonder")  # no question mark: the decision model is unsure, the Reader decides
    prompts = fake.reads()[reads_before:]
    assert len(prompts) == 3, "three tries, then the kind stands"
    assert "PREVIOUS READING WAS REJECTED" in prompts[1] and "carries writes" in prompts[1]
    assert m.version == version and m.field("claim:unobserved.exists") is None or m.field("claim:unobserved.exists").value is None
    assert "A scripted answer" in d.payload["text"], "the kind stood as ask: the Explainer answered, nothing was written"


def test_a_confident_decision_skips_the_reader_and_an_unsure_one_reaches_it():
    fake = DeskFake()
    d = Desk(fake)
    d.say(QUESTION)
    reads = len(fake.reads())
    p = d.say("plot math score by lunch")  # the decision model is sure: explore, no Reader call
    assert len(fake.reads()) == reads and p["kind"] == "chat" and "Mean math score by lunch" in p["text"] and "[artifact:" in p["text"]
    d.say("thanks, one moment")  # unsure: the Reader decides the kind
    assert len(fake.reads()) == reads + 1 and fake.calls.count("decide") >= 2
    p = d.say("why do you ask about hidden factors?")  # sure: ask, straight to the Explainer
    assert len(fake.reads()) == reads + 1 and "A scripted answer" in p["text"]
