"""The adjustment lane with a fake model and real DoWhy on the real files. No Vertex calls."""

from __future__ import annotations

import json
import re
import uuid

import pytest
from langchain_core.messages import AIMessage

from causal_agent.common.contracts import Candidate, Cited, Contrast, Handoff, Interpretation, Scope
from causal_agent.common.llm import set_llm
from causal_agent.specialists.dowhy.contracts import Contrasts, DesignAssessment, EstimatorPick, Relation, Revision
from causal_agent.specialists.dowhy.graph import compile_local

CITE = "col:test_preparation_course.note"


def students_handoff() -> Handoff:
    cols = ["math score", "test preparation course", "lunch", "parental level of education", "gender", "race/ethnicity", "reading score", "writing score"]
    return Handoff(
        family="adjustment", specialist="dowhy", supported_now=True, outcome="math score", treatment="test preparation course",
        scope=Scope(), pack_name="students",
        relevant_columns=[Candidate(column=c, reason="r", cites=[CITE]) for c in cols],
        chosen_assumption="nothing beyond lunch and parental education drove both",
        reasons=[Cited(reason="r", cites=[CITE])],
    )


def uruguay_handoff() -> Handoff:
    cols = ["Support", "Participation", "Income_Centered", "Education", "Age"]
    return Handoff(
        family="adjustment", specialist="dowhy", supported_now=True, outcome="Support", treatment="Participation",
        scope=Scope(), pack_name="gov_transfers",
        relevant_columns=[Candidate(column=c, reason="r", cites=["col:participation.note"]) for c in cols],
        chosen_assumption="forced into the adjustment lane for the negative case",
        reasons=[Cited(reason="r", cites=["col:participation.note"])],
    )


# students: how each column relates (what a careful reader of the note would answer)
STUDENT_RELATIONS = {
    "lunch": dict(affects_treatment=True, affects_outcome=True),
    "parental_level_of_education": dict(affects_treatment=True, affects_outcome=True),
    "gender": dict(affects_outcome=True),
    "race_ethnicity": dict(affects_outcome=True),
    "reading_score": dict(is_outcome_measure=True),
    "writing_score": dict(is_outcome_measure=True),
    "income_centered": dict(affects_treatment=True, affects_outcome=True),
    "education": dict(affects_outcome=True),
    "age": dict(affects_outcome=True),
}


class FakeLLM:
    def __init__(self, *, bad_cites=False, assess_script=None, pick_script=None, interpret_bad_first=False, cite=CITE):
        self.bad_cites = bad_cites
        self.assess_script = list(assess_script or [])
        self.pick_script = list(pick_script or [])
        self.interpret_bad_first = interpret_bad_first
        self.cite = cite
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
        cite = "col:nope.note" if self.bad_cites else self.cite
        if schema is Contrasts:
            levels = re.findall(r"'([^']+)'", human.split("OBSERVED LEVELS")[1].split("\n")[1])
            control = "none" if "none" in levels else levels[0]
            treated = next(v for v in levels if v != control)
            return Contrasts(items=[Contrast(control=control, treated=treated, reason="the note says completed is the course", cites=[cite])])
        if schema is Relation:
            col = re.search(r"for column '([^']+)'", human).group(1)
            flags = STUDENT_RELATIONS.get(col, {})
            claims = dict(affects_treatment=False, affects_outcome=False, affected_by_treatment=False, is_outcome_measure=False)
            claims.update(flags)
            reasons = [Cited(reason=f"{col}: {k}", cites=[cite]) for k, v in claims.items() if v]
            return Relation(column=col, reasons=reasons, **claims)
        if schema is DesignAssessment:
            if self.assess_script:
                return self.assess_script.pop(0)
            return DesignAssessment(action="proceed", reason="only soft flags; the adjustment handles them", cites=[])
        if schema is EstimatorPick:
            names = [n.strip() for n in re.search(r"NAMES YOU MAY PICK: (.*)", human).group(1).split(",")]
            name = self.pick_script.pop(0) if self.pick_script else names[0]
            return EstimatorPick(name=name, reason="ranked first and the checks allow it", cites=[])
        if schema is Interpretation:
            addresses = human.split("ADDRESSES YOU MAY CITE")[1].split("\n\n")[0].strip().splitlines()
            contrast = re.search(r"COMPARISON: (\S+)", human).group(1)
            value = float(re.search(r"\[estimate:%s\.value\] ([-\d.eE+]+)" % re.escape(contrast), human).group(1))
            bad = self.interpret_bad_first and "PREVIOUS ANSWER WAS REJECTED" not in human
            return Interpretation(contrast=contrast, answer=f"The effect is {value:.3g} points.", effect_stated=value,
                                  caveats=["bets on the router's assumption"], cites=["nope:x"] if bad else addresses[:3])
        raise AssertionError(schema)


@pytest.fixture(autouse=True)
def _restore(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_DIR", str(tmp_path / "runs"))
    yield
    set_llm(None)


def _run(fake, handoff, question="Did completing the prep course raise math scores?"):
    set_llm(fake)
    g = compile_local()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return g.invoke({"question": question, "handoff": handoff, "dataset": handoff.pack_name}, cfg)


# ------------------------------------------------------------------ tests


def test_students_happy_path():
    fake = FakeLLM()
    out = _run(fake, students_handoff())
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    d = out["design"]
    assert set(d.estimand.adjustment_set) == {"lunch", "parental_level_of_education"}
    assert {x.column for x in d.graph.excluded} == {"reading_score", "writing_score"}
    assert d.estimator == "propensity_score_stratification" and d.also_run == "linear_regression"
    primary = [e for e in out["estimates"] if not e.secondary]
    assert len(primary) == 1 and primary[0].error is None and primary[0].value > 0
    assert {e.method for e in out["estimates"]} == {"propensity_score_stratification", "linear_regression"}
    assert {x.refuter for x in out["refutations"]} == {"placebo_treatment_refuter", "random_common_cause", "data_subset_refuter"}
    placebo = next(x for x in out["refutations"] if x.refuter == "placebo_treatment_refuter")
    assert placebo.passed is True
    assert len(out["interpretations"]) == 1 and not out.get("interpret_errors")
    assert "DESIGN" in r["report"] and "ANSWER" in r["report"]
    assert fake.calls.count("Relation") == 6 and fake.calls.count("Contrasts") == 1 and fake.calls.count("EstimatorPick") == 1
    assert "DesignAssessment" in fake.calls  # students has a soft balance flag, so assess ran
    nodes = {t.node for t in out["debug"]}
    assert {"contrast", "relate:lunch", "assess", "pick_estimator", "interpret:completed_vs_none"} <= nodes
    assert (json.loads(open(f"{r['run_dir']}/design.json").read())["estimator"]) == "propensity_score_stratification"


def test_bad_cites_loop_relate_then_stop():
    fake = FakeLLM(bad_cites=True)
    out = _run(fake, students_handoff())
    r = out["specialist_result"]
    assert r["status"] == "infeasible"
    assert out["feasibility"].stage == "verify_graph"
    assert fake.calls.count("Relation") == 6 * 3  # three attempts, every worker rejected each time
    assert out.get("design") is None and not out.get("estimates")


def test_revision_is_a_delta_and_relate_not_rerun():
    script = [
        DesignAssessment(action="revise", reason="lunch is imbalanced; exclude it", cites=[],
                         revisions=[Revision(column="parental_level_of_education", change="exclude", reason="test delta", cites=[CITE])]),
        DesignAssessment(action="proceed", reason="fine now", cites=[]),
    ]
    fake = FakeLLM(assess_script=script)
    out = _run(fake, students_handoff())
    assert out["specialist_result"]["status"] == "done"
    assert out["revisions"] == 1
    assert "parental_level_of_education" in {x.column for x in out["graph"].excluded}
    assert out["design"].estimand.adjustment_set == ["lunch"]
    assert fake.calls.count("Relation") == 6  # the delta was applied by merge_graph, no worker reran


def test_uruguay_forced_stops_on_overlap():
    script = [DesignAssessment(action="stop", reason="the score decides participation outright", cites=[])]
    fake = FakeLLM(assess_script=script, cite="col:participation.note")
    out = _run(fake, uruguay_handoff(), question="Did receiving the transfer raise support for the government?")
    r = out["specialist_result"]
    assert r["status"] == "infeasible"
    f = out["feasibility"]
    assert f.stage == "assess"
    hard = {c.name for c in out["checks"] if c.level == "hard"}
    assert {"overlap", "separation"} <= hard
    assert out.get("design") is None
    assert "STOPPED AT   assess" in r["report"]


def test_hard_flag_blocks_proceed():
    script = [DesignAssessment(action="proceed", reason="ignore the flags", cites=[])] * 3
    fake = FakeLLM(assess_script=script, cite="col:participation.note")
    out = _run(fake, uruguay_handoff(), question="q")
    assert out["specialist_result"]["status"] == "infeasible"
    assert fake.calls.count("DesignAssessment") == 3
    assert out["feasibility"].stage == "assess"


def test_estimator_outside_list_is_rejected_then_accepted():
    fake = FakeLLM(pick_script=["econml_magic", "linear_regression"])
    out = _run(fake, students_handoff())
    assert out["specialist_result"]["status"] == "done"
    assert out["design"].estimator == "linear_regression"
    assert fake.calls.count("EstimatorPick") == 2


def test_interpretation_bad_cite_is_retried():
    fake = FakeLLM(interpret_bad_first=True)
    out = _run(fake, students_handoff())
    assert out["specialist_result"]["status"] == "done"
    assert fake.calls.count("Interpretation") == 2
    assert not out.get("interpret_errors")


def test_catalogues_name_real_dowhy_methods():
    from dowhy import causal_estimators, causal_refuters

    from causal_agent.specialists.dowhy.knowledge import load_checks, load_estimators, load_refuters

    for e in load_estimators():
        assert causal_estimators.get_class_object(e.dowhy.split(".", 1)[1] + "_estimator") is not None, e.name
    for r in load_refuters():
        assert causal_refuters.get_class_object(r.name) is not None, r.name
    cfg = load_checks()
    assert cfg["overlap"]["common_support_share"]["hard"] < cfg["overlap"]["common_support_share"]["soft"]
    assert cfg["balance"]["smd"]["soft"] < cfg["balance"]["smd"]["hard"]


def test_router_wires_the_real_specialist():
    from causal_agent.router.graph import graph as router_graph
    from causal_agent.specialists import SPECIALISTS

    assert "freeze_design" in SPECIALISTS["adjustment"].get_graph().nodes
    assert "relate" not in SPECIALISTS["synthetic_control"].get_graph().nodes  # still a stub
    assert len(router_graph.get_graph().nodes) == 16
