"""A run's graph written back as drafts: an edge into the treatment or the outcome per column, a measure of the outcome when
excluded as one, never over a field the person confirmed, nothing when the result has no graph."""

from __future__ import annotations

from causal_agent.desk.handoff import forced
from causal_agent.desk.relations import absorb_relations
from causal_agent.memory import store

COLS = ["math score", "test preparation course", "lunch", "gender", "reading score", "writing score"]


def _handoff(memory):
    return forced("students3", "Did completing the prep course raise math scores?", "adjustment", "math score", "test preparation course", COLS, memory=memory)


def _result(edges, excluded=(), nodes=("test_preparation_course", "math_score", "lunch", "gender")):
    return {
        "status": "done",
        "design": {
            "graph": {
                "treatment": "test_preparation_course",
                "outcome": "math_score",
                "nodes": list(nodes),
                "edges": [{"src": s, "dst": d, "cites": []} for s, d in edges],
                "excluded": [{"column": c, "why": w} for c, w in excluded],
            }
        },
    }


def test_the_graph_comes_back_as_drafts_and_never_over_a_confirmed_field():
    m = store.migrate("students3", write=False)
    m.set("col:gender.moves_outcome", False, status="confirmed", source="user:turn:4", said="gender never mattered")
    h = _handoff(m)
    v = m.version
    result = _result(
        edges=[("test_preparation_course", "math_score"), ("lunch", "test_preparation_course"), ("lunch", "math_score"), ("gender", "math_score")],
        excluded=[
            ("reading_score", "another measurement of the outcome"),
            ("writing_score", "comes after the treatment; adjusting for it would remove part of the effect"),
        ],
    )
    written = absorb_relations(m, result, h)
    assert written == ["col:lunch.feeds_treatment", "col:lunch.moves_outcome", "col:gender.feeds_treatment", "col:reading_score.measures_outcome"]
    lunch_t, lunch_y = m.field("col:lunch.feeds_treatment"), m.field("col:lunch.moves_outcome")
    assert (lunch_t.value, lunch_t.status, lunch_t.source, lunch_t.reason) == (True, "drafted", "model:relate", "the run's graph drew this edge")
    assert lunch_y.value is True and lunch_y.status == "drafted"
    gender_t = m.field("col:gender.feeds_treatment")
    assert gender_t.value is False and gender_t.status == "drafted" and gender_t.reason == "the run's graph drew no such edge"
    # the person's word stands: the gate refused the draft over the confirmed field, and the value did not move
    gender_y = m.field("col:gender.moves_outcome")
    assert gender_y.value is False and gender_y.status == "confirmed" and gender_y.source == "user:turn:4"
    reading = m.field("col:reading_score.measures_outcome")
    assert reading.value is True and reading.status == "drafted" and reading.reason == "excluded as another measurement of the outcome"
    # a column excluded for another reason, or not placed at all, gets nothing
    assert m.field("col:writing_score.measures_outcome") is None and m.field("col:writing_score.feeds_treatment") is None
    assert m.version == v + len(written)


def test_a_result_without_a_graph_writes_nothing():
    m = store.migrate("students3", write=False)
    h = _handoff(m)
    v = m.version
    assert absorb_relations(m, {}, h) == [] and absorb_relations(m, {"design": {"estimator": "x"}}, h) == [] and absorb_relations(m, {"design": None}, h) == []
    assert m.version == v


def test_the_treatment_and_the_outcome_are_never_related_to_themselves():
    m = store.migrate("students3", write=False)
    h = _handoff(m)
    written = absorb_relations(m, _result(edges=[("test_preparation_course", "math_score")], nodes=("test_preparation_course", "math_score")), h)
    assert written == [] and not any(
        a.startswith(("col:math_score.", "col:test_preparation_course.")) and a.endswith(("feeds_treatment", "moves_outcome")) for a in m.fields
    )
