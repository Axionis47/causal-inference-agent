"""The routing alone with a fake model. No Vertex calls. The memory is held in this process, never written to disk."""

from __future__ import annotations

import pytest

from causal_agent.common.contracts import Candidate, FamilyDecision, QuestionFrame, Rejection, Scope
from causal_agent.common.llm import set_llm
from causal_agent.desk.nodes import frame as F
from causal_agent.desk.route import route
from causal_agent.desk.tests.fakes import FAMILIES, QUESTION, DeskFake
from causal_agent.memory import store
from causal_agent.memory.records import Memory

GOOD = ["col:test_preparation_course.note"]
BAD = ["col:nope.note"]


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    """The memory lives in this process so mining never touches data/memory/."""
    held: dict[str, Memory] = {}

    def memory_for(name, root=None):
        if name not in held:
            held[name] = store.migrate(name, write=False)
        return held[name]

    monkeypatch.setattr(store, "memory_for", memory_for)
    monkeypatch.setattr(store, "save", lambda m, root=None: None)
    yield
    set_llm(None)


def _run(fake, dataset="students"):
    set_llm(fake)
    return route(QUESTION, dataset)


def test_students_is_mined_fitted_and_routed_to_adjustment():
    fake = DeskFake(cites=GOOD)
    out = _run(fake)
    assert len(fake.reads("doc:")) == 1 and fake.calls.count("PrefilterVote") == 0 and fake.calls.count("FamilyDecision") == 1
    assert out.frame.outcome == "math score"
    verdicts = {v.family: v for v in out.family_verdicts}
    assert set(verdicts) == set(FAMILIES)
    assert verdicts["adjustment"].admissible
    assert not verdicts["diff_in_diff"].admissible and "grain" in " ".join(n.need for n in verdicts["diff_in_diff"].needs if not n.met)
    assert not verdicts["discontinuity"].admissible
    unasked = [n for n in verdicts["adjustment"].needs if not n.met]
    assert unasked and all("not asked yet" in n.note for n in unasked)  # the beliefs: listed, never mined, never blocking here
    memory = store.memory_for("students")
    assert memory.field("claim:assignment.kind").value == "own_choice" and memory.field("claim:assignment.kind").status == "drafted"
    assert memory.field("claim:unobserved.exists") is None  # the description's belief was dropped
    h = out.handoff
    assert h.family == "adjustment" and h.specialist == "dowhy" and h.supported_now
    assert h.treated_level == "completed" and h.column("lunch").role == "depends_on" and h.column("lunch").when == "before"
    assert {c.column for c in h.relevant_columns} >= {"math score", "test preparation course", "lunch"}
    assert "CHOSEN       adjustment" in out.decision_record and "FAMILY FIT" in out.decision_record
    # the Designer wrote the brief once the gate passed; the pack bets on its sentence and the record prints it
    assert fake.calls.count("DesignBrief") == 1 and out.brief is not None and out.brief.family == "adjustment" and out.brief.road == "backdoor"
    assert h.brief == out.brief and h.chosen_assumption == out.brief.bets_on and h.resolve("design.brief.bets_on")
    assert f"BETS ON      {out.brief.bets_on}" in out.decision_record and "[design.brief.who_is_treated]" in out.decision_record
    nodes = [t.node for t in out.debug]
    assert (
        "read:doc:context" in nodes and "frame" in nodes and "decide" in nodes and "design:1" in nodes and not any(n.startswith("test_family") for n in nodes)
    )


def test_bad_citations_loop_the_gate_then_stop():
    fake = DeskFake(cites=BAD)
    out = _run(fake)
    assert fake.calls.count("FamilyDecision") == 3 and fake.calls.count("DesignBrief") == 0  # no design for a choice the gate refused
    assert all("not a pack address" in human for human in fake.humans["FamilyDecision"][1:])  # the gate's errors go round to decide
    assert out.handoff is None and out.brief is None and out.decision is not None and "FAMILY FIT" in out.decision_record


class NothingAdmissible(DeskFake):
    def answer(self, schema, human):
        if schema is FamilyDecision:
            self.calls.append(schema.__name__)
            return FamilyDecision(
                admissible=[],
                chosen="none",
                chosen_assumption="none",
                why_over_alternatives="nothing admissible",
                rejected=[Rejection(family=f, reason="unmet", cites=[]) for f in FAMILIES],
            )
        return super().answer(schema, human)


def test_the_model_may_decline_every_family_the_fit_offers():
    fake = NothingAdmissible()
    out = _run(fake)
    assert fake.calls.count("FamilyDecision") == 1
    assert out.handoff is None and "CHOSEN       none" in out.decision_record


def test_a_settled_memory_needs_no_decide_call():
    """students3 carries a full interview: one family stands, so the choice is code and the model is not asked."""
    fake = DeskFake()
    out = _run(fake, dataset="students3")
    assert len(fake.reads("doc:")) == 0 and fake.calls.count("FamilyDecision") == 0
    assert out.handoff.family == "adjustment" and out.decision.why_over_alternatives == "only admissible family"
    assert out.decision.chosen_assumption.startswith("nothing unmeasured")  # the family's words, until the Designer wrote the brief
    assert fake.calls.count("DesignBrief") == 1 and out.handoff.chosen_assumption == out.brief.bets_on != out.decision.chosen_assumption


def test_a_wide_table_is_skimmed_column_by_column(monkeypatch):
    monkeypatch.setenv("FRAME_WIDTH_BUDGET", "1")
    fake = DeskFake(cites=GOOD)
    out = _run(fake)
    assert fake.calls.count("PrefilterVote") == len(store.memory_for("students").columns)
    assert out.frame.outcome == "math score" and out.handoff.family == "adjustment"


def test_relevant_columns_not_in_the_file_are_dropped():
    memory = store.memory_for("students")
    fr = QuestionFrame(
        intent="effect_of_change",
        decision_served="d",
        outcome_candidates=[Candidate(column="math_score", reason="r", cites=["dataset.note"])],
        cause_candidates=[Candidate(column="test preparation course", reason="r", cites=["dataset.note"])],
        scope=Scope(),
        relevant_columns=[
            Candidate(column="math score", reason="r", cites=["dataset.note"]),
            Candidate(column="N/A (implicit student unit)", reason="r", cites=["dataset.note"]),
        ],
        reasons=[],
    )
    F.normalise_columns(fr, memory)
    assert [c.column for c in fr.relevant_columns] == ["math score"] and fr.outcome == "math score"
