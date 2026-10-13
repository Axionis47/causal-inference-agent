"""Explicit, source-backed lane assertions return as drafts without replacing human facts."""

from __future__ import annotations

from causal_agent.desk.handoff import forced
from causal_agent.desk.relations import absorb_relations
from causal_agent.memory import store

COLS = ["math score", "test preparation course", "lunch", "gender", "reading score", "writing score"]


def _handoff(memory):
    return forced("students3", "Did completing the prep course raise math scores?", "adjustment", "math score", "test preparation course", COLS, memory=memory)


def _result(edges, no_edges=()):
    assertions = [("edge", s, d) for s, d in edges] + [("no_edge", s, d) for s, d in no_edges]
    return {
        "status": "done",
        "ledger": {
            "revision": 1,
            "history": [],
            "entries": {
                f"rel:{i}": {
                    "id": f"rel:{i}",
                    "kind": kind,
                    "src": src,
                    "dst": dst,
                    "status": "supported",
                    "reason": "explicit source-backed proposal",
                    "evidence": [{"source": "pack", "category": "study_account", "address": f"col:{src}.when", "note": "the reported measurement account"}],
                }
                for i, (kind, src, dst) in enumerate(assertions)
            },
        },
    }


def test_explicit_assertions_are_drafts_and_cannot_replace_human_confirmation():
    m = store.migrate("students3", write=False)
    m.set("col:gender.moves_outcome", False, status="confirmed", source="user:turn:4", said="gender never mattered")
    h = _handoff(m)
    version = m.version
    written = absorb_relations(
        m, _result([("lunch", "test_preparation_course"), ("lunch", "math_score"), ("gender", "math_score")], [("gender", "test_preparation_course")]), h
    )
    assert written == ["col:lunch.feeds_treatment", "col:lunch.moves_outcome", "col:gender.feeds_treatment"]
    for address in written:
        f = m.field(address)
        assert f.status == "drafted" and f.source == "model:ledger" and "evidence col:" in f.reason
    assert m.field("col:lunch.moves_outcome").value is True
    assert m.field("col:gender.feeds_treatment").value is False
    assert m.field("col:gender.moves_outcome").value is False and m.field("col:gender.moves_outcome").status == "confirmed"
    assert m.field("col:reading_score.measures_outcome") is None
    assert m.version == version + len(written)


def test_missing_assertions_do_not_write_negative_claims():
    m = store.migrate("students3", write=False)
    h = _handoff(m)
    version = m.version
    assert absorb_relations(m, {}, h) == []
    assert absorb_relations(m, {"ledger": {"entries": {}}}, h) == []
    assert m.version == version


def test_treatment_and_outcome_do_not_receive_self_role_updates():
    m = store.migrate("students3", write=False)
    h = _handoff(m)
    assert absorb_relations(m, _result([("test_preparation_course", "math_score")]), h) == []
