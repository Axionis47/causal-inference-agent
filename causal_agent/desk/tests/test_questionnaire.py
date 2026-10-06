"""The readback as a questionnaire: every draft is one item the page can answer on its own, the desk's draft recommended and the
other legal answers beside it; a changed line comes back as `address = value`, the person's word on that field alone, and the
readback returns for the lines still drafted; a yes confirms every line."""

from __future__ import annotations

from causal_agent.desk.readback import compose_readback
from causal_agent.desk.tests.fakes import QUESTION, DeskFake
from causal_agent.desk.tests.test_journey import HELD, Desk, _held  # noqa: F401  (the autouse fixture that holds the memory in process)
from causal_agent.memory import ops


def test_each_draft_is_an_item_with_its_legal_answers_and_the_draft_recommended():
    d = Desk(DeskFake())
    d.say(QUESTION)
    d.say("let's run it")
    memory = HELD["students"]
    a = compose_readback(memory, ops.open_fields(memory, d.values["status"]) if hasattr(ops, "open_fields") else d.values["open"])
    assert a is not None and a.kind == "confirm" and a.recommended == "Yes, all right" and a.options == ["Yes, all right", "No"]
    by = {i.address: i for i in a.items}
    assert set(by) == set(a.addresses) and len(a.items) == len(a.addresses)
    kind = by["claim:assignment.kind"]
    assert kind.label == "assignment, kind" and kind.value == "own_choice" and kind.recommended == "own_choice" and "lottery" in kind.options
    panel = by["claim:grain.panel"]
    assert panel.value == "no" and panel.options == ["yes", "no"]
    when = by["col:lunch.when"]
    assert when.label == "'lunch' when" and when.value == "before" and when.options == ["before", "at", "after", "unknown"]
    what = by["claim:change.what"]
    assert what.options == [] and what.recommended == what.value


def test_a_changed_line_is_the_persons_word_and_the_readback_returns_for_the_rest():
    fake = DeskFake()
    d = Desk(fake)
    d.say(QUESTION)
    p = d.say("let's run it")
    assert p["ask"]["kind"] == "confirm" and p["ask"]["items"] and p["ask"]["recommended"] == "Yes, all right"
    n = len(p["ask"]["items"])
    reads = fake.calls.count("Reading")
    p = d.say("claim:grain.row_is = one student who sat the May exam")
    f = HELD["students"].field("claim:grain.row_is")
    assert f.value == "one student who sat the May exam" and f.status == "confirmed" and f.source.startswith("user:turn:") and fake.calls.count("Reading") == reads
    assert p["ask"]["kind"] == "confirm" and len(p["ask"]["items"]) == n - 1 and "claim:grain.row_is" not in {i["address"] for i in p["ask"]["items"]}
    p = d.say("yes, all right")
    assert p["ask"] is None or p["ask"]["kind"] != "confirm"
    assert all(HELD["students"].field(a).status == "confirmed" for a in ("claim:assignment.kind", "claim:grain.panel"))
