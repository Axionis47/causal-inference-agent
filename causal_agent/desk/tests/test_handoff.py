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
    lunch = h.column("lunch")
    assert lunch.provenance["when"].status == "confirmed" and lunch.provenance["set_by"].said == "the district sets it each September"
    text = lunch.render()
    assert '[col:lunch.set_by] set by the district · confirmed · user:turn:9 · said "the district sets it each September"' in text
    assert "[col:gender.stands_for] sex as recorded · drafted · model:infer" in h.column("gender").render()
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
    ctx = h.render_context()
    order = [
        "STORY",
        "THE PAIR",
        "THE MECHANISM",
        "TIME",
        "THE COLUMNS",
        "THE ROWS",
        "HIDDEN FACTORS",
        "HETEROGENEITY",
        "THREATS",
        "FAMILY BLOCK",
        "DESIGN BRIEF",
        "WHAT THE PERSON SAID",
    ]
    assert [ctx.index(f"\n{s}\n" if s != "STORY" else "STORY\n") for s in order] == sorted(ctx.index(f"\n{s}\n" if s != "STORY" else "STORY\n") for s in order)
    assert "[pair.treatment] the change is recorded by 'test preparation course' [col:test_preparation_course]" in ctx and h.resolve("pair.treatment")
    assert "[time.after] measured after the change: " in ctx and h.resolve("time.after") and "[col:lunch]" in ctx.split("THE COLUMNS")[1].split("THE ROWS")[0]


def test_a_forced_handoff_has_no_brief_and_a_built_one_bets_on_the_briefs_sentence():
    from causal_agent.common.contracts import Cited, DecisionMade, DesignBrief, FamilyDecision, Scope
    from causal_agent.desk import handoff as H
    from causal_agent.desk.tests.fakes import students_frame
    from causal_agent.families import registry as R

    m = _memory()
    h = forced("students3", "q", "adjustment", "math score", "test preparation course", COLS, memory=m)
    assert h.brief is None and "DESIGN BRIEF\n(no design brief)" in h.render_context() and not h.resolve("design.brief.bets_on")
    brief = DesignBrief(
        family="adjustment",
        road="backdoor",
        decisions=[
            DecisionMade(name="who_is_treated", choice="completed against none", rests_on=["claim:assignment.treated_level"], reason="the file says so")
        ],
        threats=[Cited(reason="a hidden driver of both", cites=["claim:unobserved.exists"])],
        checks=["balance on lunch"],
        bets_on="nothing beyond lunch drove both",
    )
    decision = FamilyDecision(admissible=["adjustment"], chosen="adjustment", chosen_assumption="the family's words", why_over_alternatives="only", rejected=[])
    fr = students_frame()
    fr.scope = Scope()
    h2 = H.build(question="q", frame=fr, decision=decision, family=R.family("adjustment"), memory=m, brief=brief)
    assert h2.chosen_assumption == "nothing beyond lunch drove both" and h2.brief == brief
    ctx = h2.render_context()
    assert "DESIGN BRIEF\n[design.brief.who_is_treated] completed against none (rests on [claim:assignment.treated_level]): the file says so" in ctx
    assert "[design.brief.road] backdoor" in ctx and "[design.brief.threat:1] a hidden driver of both (cites [claim:unobserved.exists])" in ctx
    assert "[design.brief.check:1] balance on lunch" in ctx and ctx.index("FAMILY BLOCK") < ctx.index("DESIGN BRIEF") < ctx.index("WHAT THE PERSON SAID")
    assert h2.resolve("design.brief.bets_on") and h2.resolve("design.brief.threat:1") and h2.resolve("design.brief")
