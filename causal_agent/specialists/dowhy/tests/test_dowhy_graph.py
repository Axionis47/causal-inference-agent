"""The adjustment lane with a fake model and real DoWhy on the real files. No Vertex calls."""

from __future__ import annotations

import json

import pandas as pd
import re
import uuid

import pytest
from langchain_core.messages import AIMessage

from causal_agent.common.contracts import Cited, Contrast, Handoff, Interpretation
from causal_agent.common.llm import set_llm
from causal_agent.desk.handoff import forced
from causal_agent.memory import store
from causal_agent.specialists.dowhy.contracts import Contrasts, DesignAssessment, EstimatorPick, Relation, Revision
from causal_agent.specialists.dowhy.graph import compile_local

CITE = "col:test_preparation_course.note"


def _memory(name):
    """The memory from the shipped claims file alone: a note mined into the memory on disk never moves a test."""
    return store.migrate(name, write=False)


def students_handoff() -> Handoff:
    cols = ["math score", "test preparation course", "lunch", "parental level of education", "gender", "race/ethnicity", "reading score", "writing score"]
    return forced("students", "Did completing the prep course raise math scores?", "adjustment", "math score", "test preparation course", cols,
                  assumption="nothing beyond lunch and parental education drove both", cite=CITE, memory=_memory("students"))


def uruguay_handoff() -> Handoff:
    cols = ["Support", "Participation", "Income_Centered", "Education", "Age"]
    return forced("gov_transfers", "Did receiving the transfer raise support for the government?", "adjustment", "Support", "Participation", cols,
                  assumption="forced into the adjustment lane for the negative case", cite="col:participation.note", memory=_memory("gov_transfers"))


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
        self.humans: list[tuple[str, str]] = []

    def humans_of(self, name: str) -> list[str]:
        return [h for n, h in self.humans if n == name]

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
        self.humans.append((schema.__name__, human))
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
            required = [a for a in human.split("ADDRESSES YOU MUST CITE")[1].split("\n\n")[0].strip().splitlines()[1:] if a and a != "(none)"]
            contrast = re.search(r"COMPARISON: (\S+)", human).group(1)
            value = float(re.search(r"\[estimate:%s\.value\] ([-\d.eE+]+)" % re.escape(contrast), human).group(1))
            bad = self.interpret_bad_first and "PREVIOUS ANSWER WAS REJECTED" not in human
            cites = ["nope:x"] if bad else list(dict.fromkeys(required + addresses[:3]))
            return Interpretation(contrast=contrast, answer=f"The effect is {value:.3g} points.", effect_stated=value,
                                  caveats=["bets on the router's assumption"] + [f"flag {a}" for a in required], cites=cites)
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
    assert len(router_graph.get_graph().nodes) == 17


# ------------------------------------------------------------------ the pack's facts end judgements


def test_pack_treated_level_settles_the_contrast_without_a_model_call():
    """students3 carries claims: the assignment names 'completed' as the treated level, so the lane never asks the model for the contrast."""
    cols = ["math score", "test preparation course", "lunch", "parental level of education", "gender", "race/ethnicity", "reading score", "writing score"]
    h = forced("students3", "Did completing the prep course raise math scores?", "adjustment", "math score", "test preparation course", cols, cite=CITE)
    assert h.treated_level == "completed" and h.control_level == "none" and h.design.kind == "adjustment"
    assert "lunch" in h.design.adjustment_candidates and h.design.unobserved_confounding is False and h.design.voluntary_uptake is True
    fake = FakeLLM()
    out = _run(fake, h)
    assert fake.calls.count("Contrasts") == 0
    c = out["contrasts"][0]
    assert c.treated == "completed" and c.control == "none" and "claim:assignment.treated_level" in c.cites
    assert out["specialist_result"]["status"] == "done", out["specialist_result"].get("feasibility")
    from causal_agent.specialists.dowhy import nodes as N

    material = N._material(out, c.key)  # the beliefs are in what the interpretation reads, with addresses it may cite
    assert "[claim:unobserved] nothing outside the file" in material and "claim:unobserved" in N._addresses(out, c.key)
    # parental education: the offer looked at it and it was fixed before, so its relation is a fact; lunch was marked 'after', so the model is asked
    assert fake.calls.count("Relation") == 5 and "relate:parental_level_of_education" not in {t.node for t in out["debug"]}
    parental = next(e for e in out["graph"].edges if e.src == "parental_level_of_education" and e.dst == "test_preparation_course")
    assert "claim:assignment.depends_on" in parental.cites
    # the person's words reach every judgement the lane makes (students3 ships no transcript, so the section is there and empty)
    assert "WHAT THE PERSON SAID" in N._frame_text(out)


# ------------------------------------------------------------------ the other roads: a hidden factor, an instrument, a mediator


def _synthetic(tmp_path, seed=0):
    """z pushes units into treatment and touches y no other way; m carries the whole effect (y = 2m + u); u drives both t and y."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = 600
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    t = (z + u + rng.normal(size=n) > 0).astype(int)
    m = t + rng.normal(size=n)
    y = 2 * m + u + rng.normal(size=n)
    csv = tmp_path / "synthetic.csv"
    pd.DataFrame({"z": z, "treated": t, "m": m, "y": y}).to_csv(csv, index=False)
    return csv


def _synthetic_memory(csv, *, hidden=True, instrument=None, mediator=None, said_none=False):
    from causal_agent.memory import ops
    from causal_agent.profile.profiler import profile

    m = ops.seed("synthetic", profile(csv), csv=str(csv))
    src = "user:turn:1"
    m.set("claim:grain.row_is", "one unit", status="confirmed", source=src)
    m.set("claim:grain.panel", False, status="confirmed", source=src)
    m.set("claim:sampling.how", "whole", status="confirmed", source=src)
    m.set("claim:change.what", "the programme", status="confirmed", source=src)
    m.set("claim:change.to_whom", "units", status="confirmed", source=src)
    m.set("claim:change.when", "last year", status="confirmed", source=src)
    m.set("claim:assignment.kind", "own_choice", status="confirmed", source=src)
    m.set("claim:assignment.rule", "units chose after an offer", status="confirmed", source=src)
    m.set("claim:assignment.treatment_column", "treated", status="confirmed", source=src)
    m.set("claim:assignment.treated_level", "1", status="confirmed", source=src)
    m.set("claim:unobserved.exists", hidden, status="confirmed", source=src, said="something we did not record drove both")
    for c, when in (("y", "after"), ("treated", "at"), ("z", "before"), ("m", "after")):
        m.set(f"col:{c}.meaning", f"{c} as recorded", status="confirmed", source=src)
        m.set(f"col:{c}.when", when, status="confirmed", source=src)
    m.set("col:m.moved_by_change", True, status="confirmed", source=src)
    if instrument:
        m.set("claim:exclusion.exists", True, status="confirmed", source=src, said="the draw pushed them in and touched nothing else")
        m.set("claim:exclusion.column", instrument, status="confirmed", source=src)
    elif said_none:
        m.set("claim:exclusion.exists", False, status="confirmed", source=src)
    if mediator:
        m.set("claim:mediator.exists", True, status="confirmed", source=src, said="it works only through m")
        m.set("claim:mediator.column", mediator, status="confirmed", source=src)
    elif said_none:
        m.set("claim:mediator.exists", False, status="confirmed", source=src)
    return m


def _synthetic_handoff(memory):
    return forced("synthetic", "Did the programme raise y?", "adjustment", "y", "treated", ["y", "treated", "z", "m"],
                  assumption="the instrument and the mediator are as the person says", cite="col:z.note", memory=memory)


def test_instrument_and_mediator_open_roads_around_a_hidden_factor(tmp_path):
    csv = _synthetic(tmp_path)
    h = _synthetic_handoff(_synthetic_memory(csv, hidden=True, instrument="z", mediator="m"))
    assert h.design.instrument == "z" and h.design.mediator == "m" and h.design.unobserved_confounding is True
    fake = FakeLLM(pick_script=["instrumental_variable"], cite="col:z.note")
    out = _run(fake, h, question="Did the programme raise y?")
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    assert fake.calls.count("Relation") == 0  # the instrument and the mediator are the person's word: facts, not judgements
    g = out["graph"]
    assert "unobserved" in g.nodes and any(e.src == "treated" and e.dst == "m" for e in g.edges) and any(e.src == "z" and e.dst == "treated" for e in g.edges)
    est = out["estimand"]
    assert "backdoor" not in est.roads and {"iv", "frontdoor"} <= set(est.roads) and est.instruments == ["z"] and est.frontdoor_set == ["m"]
    d = out["design"]
    assert d.estimator == "instrumental_variable" and d.estimand.kind == "iv" and d.estimand.adjustment_set == []
    primary = next(e for e in out["estimates"] if not e.secondary)
    assert primary.error is None and 1.2 < primary.value < 2.8  # the true effect is 2
    assert {x.refuter for x in out["refutations"]} == {"placebo_treatment_refuter", "data_subset_refuter"}


def test_a_hidden_factor_with_no_road_asks_back_about_the_mediator(tmp_path):
    csv = _synthetic(tmp_path)
    h = _synthetic_handoff(_synthetic_memory(csv, hidden=True))
    fake = FakeLLM(cite="col:z.note")
    out = _run(fake, h, question="Did the programme raise y?")
    r = out["specialist_result"]
    assert r["status"] == "ask" and r["ask"]["address"] == "claim:mediator.exists" and "through which" in r["ask"]["question"]
    assert out["feasibility"].stage == "ask" and out.get("design") is None


def test_no_road_and_the_person_says_so_takes_the_back_door_with_a_sensitivity_range(tmp_path):
    csv = _synthetic(tmp_path)
    h = _synthetic_handoff(_synthetic_memory(csv, hidden=True, said_none=True))
    fake = FakeLLM(cite="col:z.note")
    out = _run(fake, h, question="Did the programme raise y?")
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    d = out["design"]
    assert d.estimand.kind == "backdoor" and d.estimand.sensitivity_required and "unobserved" not in d.graph.nodes
    sens = next(x for x in out["refutations"] if x.refuter == "add_unobserved_common_cause")
    assert sens.kind == "sensitivity" and sens.range_low is not None and sens.range_high is not None and sens.range_low <= sens.range_high
    assert "hidden factor" in d.render()


# ------------------------------------------------------------------ the lane on the harness: the pack weighed by code


def _students3(cols=None, scope=None):
    from causal_agent.common.contracts import Scope

    cols = cols or ["math score", "test preparation course", "lunch", "parental level of education", "gender", "race/ethnicity", "reading score", "writing score"]
    return forced("students3", "Did completing the prep course raise math scores?", "adjustment", "math score", "test preparation course", cols, cite=CITE,
                  scope=scope or Scope(), memory=_memory("students3"))


def test_a_forbidden_column_never_enters_the_graph_and_the_flags_are_cited():
    """students3 marks the two other scores 'after': the pack forbids them. A model that calls one of them a plain parent of the
    outcome is overruled by code; the other it calls a measure of the outcome, which is excluded first."""
    h = _students3()
    assert {"reading_score", "writing_score"} <= set(h.design.forbidden)

    class Fake(FakeLLM):
        def answer(self, schema, human):
            if schema is Relation and "for column 'writing_score'" in human:
                return Relation(column="writing_score", affects_treatment=False, affects_outcome=True, affected_by_treatment=False, is_outcome_measure=False,
                                reasons=[Cited(reason="writing: moves the outcome", cites=[CITE])])
            return super().answer(schema, human)

    fake = Fake()
    out = _run(fake, h)
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    why = {x.column: x.why for x in out["graph"].excluded}
    assert "design.forbidden" in why["writing_score"] and "measurement of the outcome" in why["reading_score"]
    assert not any(e.src in ("reading_score", "writing_score") for e in out["graph"].edges)
    # the case reached every relate prompt: the settled block for a column the pack partly settles, the pack's cards and probes
    human = next(m for m in fake.humans_of("Relation") if "for column 'gender'" in m)
    assert "SETTLED BY THE PACK" in human and "affected_by_treatment = false [col:gender.when]" in human and "[probe:adjustment" in human and "[dataset.profile.rows]" in human
    # the interpretation had to cite every flag, and the artifacts carry the case and the checks
    assert not out.get("interpret_errors")
    arts = json.loads(open(f"{r['run_dir']}/artifacts.json").read())
    assert arts["case"]["facts"]["col:gender.when"] == "before" and any(c["name"] == "arms" for c in arts["checks"]) and arts["design"]["estimator"]
    ids = [f["id"] for f in json.loads(open(f"{r['run_dir']}/figures.json").read())]
    assert "effect_completed_vs_none" in ids and r["figures"] == ids


def test_relate_is_skipped_when_the_pack_settles_every_claim():
    """A column the person said the change moved and that measures nothing: affected, and open only on affects_outcome, so the model is asked;
    one it called a measure of the outcome, or the offer looked at and was fixed before, is never asked about."""
    h = _students3()
    fake = FakeLLM()
    out = _run(fake, h)
    asked = {t.node.split(":", 1)[1] for t in out["debug"] if t.node.startswith("relate:")}
    assert "parental_level_of_education" not in asked and "gender" in asked
    from causal_agent.specialists.dowhy import nodes as N

    claims, cites = N.settled_claims(h, "gender", out["case"])
    assert claims == {"affected_by_treatment": False} and cites == {"affected_by_treatment": "col:gender.when"}
    claims, _ = N.settled_claims(h, "lunch", out["case"])
    assert claims.get("affects_treatment") is True  # the offer depended on it, whatever the timing says


def test_the_filter_is_applied_by_code_and_a_prose_filter_is_a_recorded_decline():
    from causal_agent.common.contracts import Scope

    out = _run(FakeLLM(), _students3(scope=Scope(population_filter="gender == female")))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    assert out["check_facts"]["intake"] == {"filter": "gender == female", "rows_before": 1000, "rows_after": 518}
    assert len(pd.read_csv(out["table_path"])) == 518 and out["declines"] == []
    out = _run(FakeLLM(), _students3(scope=Scope(population_filter="students who sat the May exam")))
    r = out["specialist_result"]
    assert r["status"] == "done" and [d["check"] for d in r["declines"]] == ["intake.filter_unparsed"]
    assert "DISAGREEMENTS WITH THE PACK" in r["report"] and "[decline:load.scope_population_filter]" in r["report"]


def test_a_pack_named_mediator_outside_the_frame_is_loaded_and_a_confirmed_mediator_without_a_column_asks_for_it(tmp_path):
    csv = _synthetic(tmp_path)
    m = _synthetic_memory(csv, hidden=True, mediator="m")
    h = forced("synthetic", "Did the programme raise y?", "adjustment", "y", "treated", ["y", "treated", "z"], cite="col:z.note", memory=m)
    assert h.design.mediator == "m" and "m" not in [c.column for c in h.relevant_columns]
    fake = FakeLLM(cite="col:z.note")
    out = _run(fake, h, question="Did the programme raise y?")
    assert "m" in out["columns"] and any(e.src == "treated" and e.dst == "m" for e in out["graph"].edges)
    assert out["specialist_result"]["status"] == "done", out["specialist_result"].get("feasibility")
    # the person says there is a mediator but not which column: the lane asks for the column, not whether it exists
    m2 = _synthetic_memory(csv, hidden=True)
    m2.set("claim:mediator.exists", True, status="confirmed", source="user:turn:2", said="it works through something we measured")
    out = _run(FakeLLM(cite="col:z.note"), _synthetic_handoff(m2), question="Did the programme raise y?")
    r = out["specialist_result"]
    assert r["status"] == "ask" and r["ask"]["address"] == "claim:mediator.column" and "Which column" in r["ask"]["question"] and r["ask"]["stage"] == "identify"


def test_sensitivity_survives_a_revise_loop_and_a_repick_does_not_duplicate_estimates(tmp_path):
    csv = _synthetic(tmp_path)
    h = _synthetic_handoff(_synthetic_memory(csv, hidden=True, said_none=True))
    # z read as a parent of both puts it in the adjustment set; its imbalance is flagged; the assessment revises it out, then proceeds
    script = [DesignAssessment(action="revise", reason="z is too imbalanced to adjust for", cites=[], revisions=[Revision(column="z", change="exclude", reason="test delta", cites=["col:z.note"])]),
              DesignAssessment(action="proceed", reason="fine", cites=[])]

    class Fake(FakeLLM):
        def answer(self, schema, human):
            if schema is Relation and "for column 'z'" in human:
                return Relation(column="z", affects_treatment=True, affects_outcome=True, affected_by_treatment=False, is_outcome_measure=False,
                                reasons=[Cited(reason="z: fed the decision and moves y", cites=["col:z.note"])])
            return super().answer(schema, human)

    fake = Fake(assess_script=script, cite="col:z.note", pick_script=["econml_magic", "linear_regression"])
    out = _run(fake, h, question="Did the programme raise y?")
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    assert out["revisions"] == 1 and out["design"].estimand.sensitivity_required and "unobserved" not in out["design"].graph.nodes
    assert any(x.refuter == "add_unobserved_common_cause" for x in out["refutations"])
    assert "check:all.belief.unobserved" in {c.address for c in out["checks"]}
    keys = [(e.contrast, e.method) for e in out["estimates"]]
    assert len(keys) == len(set(keys))


def test_spillover_the_person_kept_is_a_flag_the_interpretation_cites(tmp_path):
    csv = _synthetic(tmp_path)
    m = _synthetic_memory(csv, hidden=False)
    m.set("claim:spillover.possible", True, status="confirmed", source="user:turn:3", said="they talk to each other")
    fake = FakeLLM(cite="col:z.note")
    out = _run(fake, _synthetic_handoff(m), question="Did the programme raise y?")
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    flag = next(c for c in out["checks"] if c.name == "belief.spillover")
    assert flag.level == "soft" and "carries part of the effect" in flag.detail
    assert "check:all.belief.spillover" in out["interpretations"][0].cites and not out.get("interpret_errors")
    assert "DesignAssessment" in fake.calls  # a flag from the person's word is a flag the assessment answers


def test_the_run_leaves_its_graph_and_its_balance_as_figures():
    fake = FakeLLM()
    out = _run(fake, _students3())
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    figs = json.loads(open(f"{r['run_dir']}/figures.json").read())
    assert [f["id"] for f in figs] == ["causal_graph", "balance_completed_vs_none", "effect_completed_vs_none"] and r["figures"] == [f["id"] for f in figs]
    g = figs[0]
    roles = {n["id"]: n["role"] for n in g["nodes"]}
    assert roles["test_preparation_course"] == "treatment" and roles["math_score"] == "outcome" and roles["parental_level_of_education"] == "confounder" and roles["reading_score"] == "excluded"
    assert any(e["src"] == "parental_level_of_education" and e["dst"] == "test_preparation_course" and "claim:assignment.depends_on" in e["cites"] for e in g["edges"])
    b = figs[1]
    assert [s["name"] for s in b["series"]] == ["before adjustment", "after weighting on the score"] and set(b["series"][0]["x"]) == {"lunch", "parental level of education"}
    assert all(a is not None and a < 0.3 for a in b["series"][1]["y"])
    assert not any(x["check"] == "figure.check" for x in r["declines"])
