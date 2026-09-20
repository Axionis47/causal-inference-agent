"""A model answer against the pack: a contradiction of a settled fact needs a cite to a contested address."""

from __future__ import annotations

from causal_agent.common.contracts import ColumnBrief, Provenance
from causal_agent.desk.handoff import forced
from causal_agent.lane import case as C
from causal_agent.lane import verify as V
from causal_agent.memory import store

RULES: list[V.Rule] = [
    ("affected_by_treatment", True, "when", ("before",)),
    ("usable_as_control", True, "moved_by_change", (True,)),
]


def pack():
    h = forced("cigar", "q", "diff_in_diff", "sales", "state", ["state", "year", "sales", "pop"], memory=store.migrate("cigar", write=False))
    h.columns = [ColumnBrief(name="pop", key="pop", when="before", moved_by_change=True,
                             provenance={"when": Provenance(status="confirmed", source="user:turn:2"), "moved_by_change": Provenance(status="confirmed", source="user:turn:2")})]
    return h


def test_a_contradicting_claim_is_rejected_without_a_cite_and_accepted_with_one():
    h = pack()
    case = C.weigh(h, {})
    errs = V.contradictions({"affected_by_treatment": True, "usable_as_control": False}, "pop", case, RULES, ["col:pop.note"], h)
    assert len(errs) == 1 and "[col:pop.when] = 'before'" in errs[0]
    assert V.contradictions({"affected_by_treatment": False, "usable_as_control": False}, "pop", case, RULES, [], h) == []
    # the pack itself marks the field contested: citing it is enough
    h.contradictions = ["col:pop.when"]
    assert V.contradictions({"affected_by_treatment": True}, "pop", case, RULES, ["col:pop.when"], h) == []
    assert len(V.contradictions({"affected_by_treatment": True}, "pop", case, RULES, ["col:pop.note"], h)) == 1
    # a drafted field is not a fact and cannot be contradicted
    h.columns[0].provenance["when"].status = "drafted"
    assert V.contradictions({"affected_by_treatment": True}, "pop", C.weigh(h, {}), RULES, [], h) == []


def test_cites_must_resolve():
    h = pack()
    assert V.cites_resolve(["col:pop.note", "col:nope.note"], h) == ["citation 'col:nope.note' does not resolve in the pack"]
