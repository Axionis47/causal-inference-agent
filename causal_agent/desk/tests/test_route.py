"""The routing graph with a fake model. No Vertex calls. The memory is held in this process, never written to disk."""

from __future__ import annotations

import re
import uuid

import pytest
from langchain_core.messages import AIMessage

from causal_agent.common.contracts import Candidate, FamilyDecision, PrefilterVote, QuestionFrame, Rejection, Scope
from causal_agent.common.llm import set_llm
from causal_agent.desk.nodes import frame as F
from causal_agent.desk.route import compile_local
from causal_agent.knowledge import load_registry
from causal_agent.memory import store
from causal_agent.memory.claims import ClaimUpdate, Extraction, FieldValue
from causal_agent.memory.records import Memory

FAMILIES = [f.name for f in load_registry()]


def U(claim_kind, cites, column=None, **values):
    return ClaimUpdate(
        kind=claim_kind, column=column, reason="scripted", cites=list(cites), values=[FieldValue(name=k, value=str(v)) for k, v in values.items()]
    )


def students_doc_updates(cite="doc:context"):
    """What a careful reading of the students note yields: the dataset kinds and one measured claim per column."""
    ups = [
        U("grain", [cite], row_is="one student's exam results", panel="false"),
        U("sampling", [cite], how="whole", detail="every student who sat the exam"),
        U("change", [cite], what="a six-week test preparation course", to_whom="students at the school", when="the six weeks before the May 2026 exam"),
        U(
            "assignment",
            [cite],
            kind="own_choice",
            rule="offered first by lunch status and parental education, then open to anyone who asked",
            depends_on="lunch, parental level of education",
            treatment_column="test preparation course",
            treated_level="completed",
        ),
        U("unobserved", [cite], exists="false"),  # a description never sets a belief: this one must be dropped by mine
    ]
    cols = {
        "gender": "before",
        "race/ethnicity": "before",
        "parental level of education": "before",
        "lunch": "before",
        "test preparation course": "at",
        "math score": "after",
        "reading score": "after",
        "writing score": "after",
    }
    for c, when in cols.items():
        ups.append(U("measured", [cite], column=c, meaning=f"{c} as recorded", when=when))
    return ups


class FakeLLM:
    def __init__(self, bad_cites: bool = False):
        self.bad_cites = bad_cites
        self.calls: list[str] = []

    def with_structured_output(self, schema, include_raw=False):
        fake = self

        class R:
            def invoke(self_, messages):
                human = messages[-1][1]
                parsed = fake.answer(schema, human)
                raw = AIMessage(
                    content=[{"type": "thinking", "thinking": f"thinking about {schema.__name__}"}, "{}"],
                    usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30, "output_token_details": {"reasoning": 7}},
                )
                return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return R()

    def answer(self, schema, human):
        self.calls.append(schema.__name__)
        cite = ["col:nope.note"] if self.bad_cites else ["col:test_preparation_course.note"]
        if schema is Extraction:
            return Extraction(updates=students_doc_updates(), confirmed=[], notes=[])
        if schema is QuestionFrame:
            c = lambda col, why, a: Candidate(column=col, reason=why, cites=[a])  # noqa: E731
            return QuestionFrame(
                intent="effect_of_change",
                decision_served="whether to keep running the course",
                outcome_candidates=[c("math score", "the question asks about math scores", "col:math_score.note")],
                cause_candidates=[c("test preparation course", "a decision by the counsellor", "col:test_preparation_course.note")],
                scope=Scope(),
                relevant_columns=[
                    c("math score", "outcome", "col:math_score.note"),
                    c("test preparation course", "cause", "col:test_preparation_course.note"),
                    c("lunch", "the course note says places depended on it", "col:test_preparation_course.note"),
                    c("parental level of education", "same", "col:test_preparation_course.note"),
                ],
                reasons=[],
            )
        if schema is PrefilterVote:
            return PrefilterVote(column="x", relevant=True, reason="r", cites=["dataset.note"])
        if schema is FamilyDecision:
            admissible = re.findall(r"^(\w+): ADMISSIBLE", human, re.M)
            return FamilyDecision(
                admissible=["adjustment"],
                chosen="adjustment",
                chosen_assumption="nothing else drove both",
                why_over_alternatives="the only family whose needs the memory meets without a belief still to ask"
                if len(admissible) > 1
                else "only admissible family",
                rejected=[Rejection(family=f, reason="a need is unmet", cites=cite) for f in FAMILIES if f != "adjustment"],
                cites=cite,
            )
        raise AssertionError(schema)


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    """Specialists are stubbed; the memory lives in this process so mining never touches data/memory/."""
    from causal_agent.families import registry as R

    stubs = {name: R.stub_lane(name, "stub", True) for name in R.REGISTRY}
    monkeypatch.setattr(R, "lanes", lambda: stubs)
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
    g = compile_local()
    return g.invoke({"question": "Did completing the prep course raise math scores?", "dataset": dataset}, {"configurable": {"thread_id": str(uuid.uuid4())}})


def test_students_is_mined_fitted_and_routed_to_adjustment():
    fake = FakeLLM()
    out = _run(fake)
    assert fake.calls.count("Extraction") == 1 and fake.calls.count("PrefilterVote") == 0
    assert out["frame"].outcome == "math score"
    verdicts = {v.family: v for v in out["family_verdicts"]}
    assert set(verdicts) == set(FAMILIES)
    assert verdicts["adjustment"].admissible
    assert not verdicts["diff_in_diff"].admissible and "grain" in " ".join(n.need for n in verdicts["diff_in_diff"].needs if not n.met)
    assert not verdicts["discontinuity"].admissible
    unasked = [n for n in verdicts["adjustment"].needs if not n.met]
    assert unasked and all("not asked yet" in n.note for n in unasked)  # the beliefs: listed, never mined, never blocking here
    memory = store.memory_for("students")
    assert memory.field("claim:assignment.kind").value == "own_choice" and memory.field("claim:assignment.kind").status == "drafted"
    assert memory.field("claim:unobserved.exists") is None  # the description's belief was dropped
    assert out["gate_errors"] == []
    h = out["handoff"]
    assert h.family == "adjustment" and h.specialist == "dowhy" and h.supported_now
    assert h.treated_level == "completed" and h.brief("lunch").role == "depends_on" and h.brief("lunch").when == "before"
    assert {c.column for c in h.relevant_columns} >= {"math score", "test preparation course", "lunch"}
    assert out["specialist_result"]["status"] == "stub"
    assert "CHOSEN       adjustment" in out["decision_record"] and "FAMILY FIT" in out["decision_record"]
    nodes = [t.node for t in out["debug"]]
    assert "mine" in nodes and "frame" in nodes and "decide" in nodes and not any(n.startswith("test_family") for n in nodes)


def test_bad_citations_loop_the_gate_then_stop():
    fake = FakeLLM(bad_cites=True)
    out = _run(fake)
    assert fake.calls.count("FamilyDecision") == 3 and out["decide_attempts"] == 3
    assert any("not a pack address" in e for e in out["gate_errors"])
    assert out.get("handoff") is None and out.get("specialist_result") is None


class NothingAdmissible(FakeLLM):
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
    assert out["gate_errors"] == [] and out.get("handoff") is None and out.get("specialist_result") is None
    assert "CHOSEN       none" in out["decision_record"]


def test_a_settled_memory_needs_no_decide_call():
    """students3 carries a full interview: one family stands, so the choice is code and the model is not asked."""
    fake = FakeLLM()
    out = _run(fake, dataset="students3")
    assert fake.calls.count("Extraction") == 0 and fake.calls.count("FamilyDecision") == 0
    assert out["handoff"].family == "adjustment" and out["decision"].why_over_alternatives == "only admissible family"
    assert out["decision"].chosen_assumption.startswith("nothing unmeasured")


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
