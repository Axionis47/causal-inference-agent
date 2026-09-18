"""Router graph with a fake model. No Vertex calls."""

import re
import uuid

import pytest
from langchain_core.messages import AIMessage

from causal_agent.common.contracts import Candidate, FamilyDecision, FamilyVerdict, NeedCheck, PrefilterVote, QuestionFrame, Rejection, Scope
from causal_agent.common.llm import set_llm
from causal_agent.knowledge import load_registry
from causal_agent.router.graph import compile_local

FAMILIES = [f.name for f in load_registry()]
NEEDS = {f.name: f.needs for f in load_registry()}


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
                raw = AIMessage(content=[{"type": "thinking", "thinking": f"thinking about {schema.__name__}"}, "{}"],
                                usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30, "output_token_details": {"reasoning": 7}})
                return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return R()

    def answer(self, schema, human):
        self.calls.append(schema.__name__)
        cite = ["col:nope.note"] if self.bad_cites else ["col:test_preparation_course.note"]
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
        if schema is FamilyVerdict:
            fam = re.search(r"family: (\w+)", human).group(1)
            ok = fam == "adjustment"
            return FamilyVerdict(family=fam, admissible=ok, needs=[NeedCheck(need=n, met=ok, cites=cite if ok else [], note="fake") for n in NEEDS[fam]])
        if schema is FamilyDecision:
            return FamilyDecision(
                admissible=["adjustment"], chosen="adjustment", chosen_assumption="nothing else drove both",
                why_over_alternatives="only admissible family",
                rejected=[Rejection(family=f, reason="a need is unmet", cites=cite) for f in FAMILIES if f != "adjustment"],
                cites=cite,
            )
        raise AssertionError(schema)


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    # The router's tests grade the router. Specialists are stubbed so the fake model never sees their schemas.
    from causal_agent import specialists as S
    from causal_agent.router import graph as G

    stubs = {name: S._stub(name, "stub", True) for name in S.SPECIALISTS}
    monkeypatch.setattr(G, "SPECIALISTS", stubs)
    yield
    set_llm(None)


def _run(fake):
    set_llm(fake)
    g = compile_local()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return g.invoke({"question": "Did completing the prep course raise math scores?", "dataset": "students"}, cfg)


def test_students_routes_to_adjustment():
    fake = FakeLLM()
    out = _run(fake)
    assert out["frame"].outcome == "math score"
    assert len(out["family_verdicts"]) == len(FAMILIES)          # one worker per family
    assert fake.calls.count("PrefilterVote") == 0                 # narrow table: no prefilter
    assert out["gate_errors"] == []
    h = out["handoff"]
    assert h.family == "adjustment" and h.specialist == "dowhy" and h.supported_now
    assert {c.column for c in h.relevant_columns} >= {"math score", "test preparation course", "lunch"}
    assert out["specialist_result"]["status"] == "stub"
    assert out["specialist_result"]["relevant_columns"][0] == "math score"
    assert "CHOSEN       adjustment" in out["decision_record"]
    # thoughts captured once per model call, never used for routing
    nodes = [t.node for t in out["debug"]]
    assert "frame" in nodes and "decide" in nodes and f"test_family:adjustment" in nodes
    assert all(t.thinking_tokens == 7 for t in out["debug"])
    assert "MODEL THOUGHTS" in out["decision_record"]


def test_bad_citations_loop_the_gate_then_stop():
    fake = FakeLLM(bad_cites=True)
    out = _run(fake)
    assert fake.calls.count("FamilyDecision") == 3
    assert out["decide_attempts"] == 3
    assert any("not a pack address" in e for e in out["gate_errors"])
    assert out.get("handoff") is None
    assert out.get("specialist_result") is None


class NothingAdmissible(FakeLLM):
    def answer(self, schema, human):
        if schema is FamilyVerdict:
            fam = re.search(r"family: (\w+)", human).group(1)
            self.calls.append(schema.__name__)
            return FamilyVerdict(family=fam, admissible=False, needs=[NeedCheck(need=n, met=False, note="fake") for n in NEEDS[fam]])
        if schema is FamilyDecision:
            self.calls.append(schema.__name__)
            return FamilyDecision(admissible=[], chosen="none", chosen_assumption="none", why_over_alternatives="nothing admissible",
                                  rejected=[Rejection(family=f, reason="unmet", cites=[]) for f in FAMILIES])
        return super().answer(schema, human)


def test_no_admissible_family_stops_honestly():
    fake = NothingAdmissible()
    out = _run(fake)
    assert fake.calls.count("FamilyDecision") == 1          # gate accepts 'none' first time
    assert out["gate_errors"] == []
    assert out.get("handoff") is None
    assert out.get("specialist_result") is None
    assert "CHOSEN       none" in out["decision_record"]


def test_relevant_columns_not_in_the_file_are_dropped():
    from causal_agent.common.contracts import Candidate, QuestionFrame, Scope
    from causal_agent.intake.datasets import load_dataset_pack
    from causal_agent.router import nodes as N

    pack = load_dataset_pack("students")
    fr = QuestionFrame(intent="effect_of_change", decision_served="d", outcome_candidates=[Candidate(column="math_score", reason="r", cites=["dataset.note"])],
                       cause_candidates=[Candidate(column="test preparation course", reason="r", cites=["dataset.note"])], scope=Scope(),
                       relevant_columns=[Candidate(column="math score", reason="r", cites=["dataset.note"]), Candidate(column="N/A (implicit student unit)", reason="r", cites=["dataset.note"])],
                       reasons=[])
    N._normalise_columns(fr, pack)
    assert [c.column for c in fr.relevant_columns] == ["math score"] and fr.outcome == "math score"
