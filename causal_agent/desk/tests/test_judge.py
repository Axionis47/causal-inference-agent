"""The judge reads the pack at the ready moment and is held by code: a verdict with the gaps it needs, every gap on a field the
pack carries with a cite that resolves, no number the pack lacks. Its reading is said in the ready moment and shown on the
handoff view; it never blocks the word."""

from __future__ import annotations

from causal_agent.desk import judge as JG
from causal_agent.desk.contracts import Gap, Readiness
from causal_agent.desk.tests.fakes import QUESTION, DeskFake
from causal_agent.desk.tests.test_journey import HELD, Desk, _held  # noqa: F401  (the autouse fixture that holds the memory in process)


def _at_ready(fake: DeskFake) -> Desk:
    d = Desk(fake)
    d.say(QUESTION)
    d.to_ready()
    return d


def test_the_gate_holds_the_verdict_to_its_gaps_and_the_gaps_to_the_pack():
    d = _at_ready(DeskFake())
    pack, memory, probes = d.values["preview"], HELD["students"], d.values["probes"]
    ok = Readiness(
        verdict="risky",
        summary="The offer rule rests on the person's word.",
        gaps=[Gap(address="claim:assignment.kind", reason="said once", cites=["claim:assignment.kind"])],
    )
    assert JG.check(ok, pack, memory, probes) == []
    assert JG.check(Readiness(verdict="ready", summary="Fine.", gaps=ok.gaps), pack, memory, probes) == [
        "verdict is ready but gaps are named; a named gap makes it risky or not_ready"
    ]
    assert JG.check(Readiness(verdict="not_ready", summary="Thin.", gaps=[]), pack, memory, probes) == [
        "verdict is not_ready but no gap is named; name at least one field, with a cite"
    ]
    bad = Readiness(verdict="risky", summary="The 999 rows worry me.", gaps=[Gap(address="claim:nope.x", reason="r", cites=["col:nope.note"])])
    errors = JG.check(bad, pack, memory, probes)
    assert errors == [
        "gap names 'claim:nope.x', which is not a field the pack carries",
        "gap on claim:nope.x cites 'col:nope.note', which is not an address in the pack",
        "'The 999 rows worry me.' states [999.0], which the pack does not hold",
    ]


def test_the_ready_moment_says_the_judges_line_and_a_refused_reading_is_re_prompted_then_dropped():
    risky = Readiness(
        verdict="risky",
        summary="The offer rule rests on the person's word alone.",
        gaps=[Gap(address="claim:assignment.kind", reason="said once, never checked", cites=["claim:assignment.kind"])],
    )
    fake = DeskFake(readiness=[Readiness(verdict="ready", summary="x", gaps=risky.gaps), risky])  # the first is refused by the gate
    d = _at_ready(fake)
    p = d.payload
    assert (
        p["kind"] == "ready"
        and "A second reading says risky: The offer rule rests on the person's word alone. said once, never checked [claim:assignment.kind; claim:assignment.kind]"
        in p["text"]
    )
    assert d.values["readiness"].verdict == "risky" and fake.calls.count("Readiness") == 2
    assert "PREVIOUS ATTEMPT FAILED THESE CHECKS" in fake.humans["Readiness"][1] and "verdict is ready but gaps are named" in fake.humans["Readiness"][1]
    d.say("yes, run it")  # advice never blocks
    assert d.values["phase"] == "after" and len(d.values["runs"]) == 1
    # three refusals: no verdict, the ready moment goes on without one
    fake2 = DeskFake(readiness=[Readiness(verdict="not_ready", summary="x", gaps=[])] * 3)
    d2 = _at_ready(fake2)
    assert (
        d2.payload["kind"] == "ready"
        and d2.values["readiness"] is None
        and "A second reading" not in d2.payload["text"]
        and fake2.calls.count("Readiness") == 3
    )
