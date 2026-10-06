"""A message of `address = value` lines is the person's word on those fields: written confirmed from their turn, never read by a
model, refused by the memory's gate with the reason in the reply."""

from __future__ import annotations

from causal_agent.desk import edits as ED
from causal_agent.desk.tests.fakes import QUESTION, DeskFake
from causal_agent.desk.tests.test_journey import HELD, Desk, _held  # noqa: F401  (the autouse fixture that holds the memory in process)


def test_only_a_message_of_edit_lines_is_edits():
    assert ED.edits_in("claim:change.when = March 2026") == [("claim:change.when", "March 2026")]
    assert ED.edits_in("col:lunch.when = before\n\nclaim:grain.panel=false ") == [("col:lunch.when", "before"), ("claim:grain.panel", "false")]
    assert ED.edits_in("the cutoff is zero") is None
    assert ED.edits_in("claim:change.when = March 2026\nand run it") is None
    assert ED.edits_in("   ") is None


def test_an_edit_at_the_ready_moment_is_written_as_the_persons_word_and_the_ready_moment_returns():
    fake = DeskFake()
    d = Desk(fake)
    d.say(QUESTION)
    d.to_ready()
    reads, decides = fake.calls.count("Reading"), fake.calls.count("decide")
    p = d.say("claim:change.when = March 2026")
    f = HELD["students"].field("claim:change.when")
    assert f.value == "March 2026" and f.status == "confirmed" and f.source.startswith("user:turn:") and f.said == "claim:change.when = March 2026"
    assert fake.calls.count("Reading") == reads and fake.calls.count("decide") == decides  # no model read the edit
    assert p["kind"] == "ready" and d.values["preview"].facts["claim:change.when"].value == "March 2026"
    assert "Noted" in p["text"] and "claim:change.when" in p["text"]
    v = HELD["students"].version
    p = d.say("claim:nope.what = x\nclaim:grain.panel = maybe")
    assert HELD["students"].version == v and p["kind"] == "ready"
    assert "Not written: claim:nope.what: unknown claim kind 'nope'; claim:grain.panel: grain.panel must be true or false, not 'maybe'." in p["text"]
