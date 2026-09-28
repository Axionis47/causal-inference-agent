"""The checkpoint serialiser keeps every contract it is told about: a class it does not know comes back as a dict, and
the server would then fail on attribute access, so the allowlist is tested here rather than discovered live."""

from __future__ import annotations

import inspect

from causal_agent.common.contracts import DecisionMade, Decline, DesignBrief, Handoff, RunRecord
from causal_agent.desk.contracts import AfterReply, Ask, FieldUpdate, Reading
from causal_agent.desk.graph import _CONTRACTS, serde
from causal_agent.desk.handoff import forced
from causal_agent.memory import store
from causal_agent.memory.records import Memory


def roundtrip(obj):
    return serde.loads_typed(serde.dumps_typed(obj))


def test_every_listed_class_is_importable_by_its_own_path():
    for entry in _CONTRACTS:
        assert inspect.isclass(entry) and entry.__module__.startswith("causal_agent."), entry


def test_the_records_the_desk_checkpoints_come_back_as_themselves():
    rec = RunRecord(index=1, dataset="students3", question="q", family="adjustment", specialist="dowhy", status="done", effect=1.5)
    back = roundtrip(rec)
    assert isinstance(back, RunRecord) and back.effect == 1.5 and back.family == "adjustment"
    ask = Ask(addresses=["col:lunch.when"], kind="story", text="Tell me the story.", decision="")
    assert isinstance(roundtrip(ask), Ask)
    reading = Reading(updates=[FieldUpdate(address="col:lunch.when", value="before", said="at enrolment")], confirms=["claim:assignment.kind"])
    back = roundtrip(reading)
    assert isinstance(back, Reading) and isinstance(back.updates[0], FieldUpdate) and back.confirms == ["claim:assignment.kind"]
    assert (
        Reading in _CONTRACTS
        and AfterReply in _CONTRACTS
        and isinstance(roundtrip(AfterReply(kind="answer", text="t", cites=["family:adjustment"])), AfterReply)
    )
    d = Decline(stage="load", kind="declined", about="scope.window", pack_value="x", took=None, reason="r", check="intake.window_unparsed")
    assert isinstance(roundtrip(d), Decline)
    h = forced("students3", "q", "adjustment", "math score", "test preparation course", ["lunch"], memory=store.migrate("students3", write=False))
    back = roundtrip(h)
    assert isinstance(back, Handoff) and back.design is not None and back.design.kind == "adjustment"
    assert isinstance(roundtrip(store.migrate("students3", write=False)), Memory)
    b = DesignBrief(
        family="adjustment", road="backdoor", decisions=[DecisionMade(name="road", choice="c", rests_on=["change:1.note"], reason="r")], bets_on="b"
    )
    back = roundtrip(b)
    assert isinstance(back, DesignBrief) and isinstance(back.decisions[0], DecisionMade) and back.road == "backdoor"
