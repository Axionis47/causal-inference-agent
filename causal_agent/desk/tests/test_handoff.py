"""The pack projected from a memory: every field with how it was settled, the open fields, the versions."""

from __future__ import annotations

from causal_agent.desk.handoff import forced
from causal_agent.memory import store

COLS = ["math score", "test preparation course", "lunch", "parental level of education", "gender"]


def _memory():
    return store.migrate("students3", write=False)


def test_brief_renders_every_field_with_status_source_and_said():
    m = _memory()
    m.set("col:lunch.set_by", "the district", status="confirmed", source="user:turn:9", said="the district sets it each September")
    m.set("col:gender.stands_for", "sex as recorded", status="drafted", source="model:infer")
    h = forced("students3", "q", "adjustment", "math score", "test preparation course", COLS, memory=m)
    lunch = h.brief("lunch")
    assert lunch.provenance["when"].status == "confirmed" and lunch.provenance["set_by"].said == "the district sets it each September"
    text = lunch.render()
    assert "[col:lunch.set_by] set by the district · confirmed · user:turn:9 · said \"the district sets it each September\"" in text
    assert "[col:gender.stands_for] sex as recorded · drafted · model:infer" in h.brief("gender").render()
    assert h.memory_version == m.version and h.design_id == 0
    assert h.resolve("col:lunch.set_by") and h.resolve("col:gender.stands_for")


def test_unknowns_and_contradictions_reach_the_lane():
    m = _memory()
    m.set("claim:sampling.detail", None, status="unknown", source="user:turn:4")
    f = m.set("col:lunch.when", "after", status="contradiction", source="user:turn:6", said="no, it really was after")
    h = forced("students3", "q", "adjustment", "math score", "test preparation course", COLS, memory=m)
    assert h.unknowns == ["claim:sampling.detail"] and h.contradictions == ["col:lunch.when"]
    words = h.render_words()
    assert "CONTRADICTION" in words and "[col:lunch.when]" in words and "UNKNOWN" in words
    assert "[said:6]" in words and f.said in words
    assert "CONTRADICTION" in h.render_context()
