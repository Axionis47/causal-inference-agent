"""The adjustment lane with a fake model and real DoWhy on the real files. No Vertex calls."""

from __future__ import annotations

import json
import re
import uuid

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from causal_agent.common.contracts import Cited, Contrast, Handoff, Interpretation
from causal_agent.common.llm import set_llm
from causal_agent.desk.handoff import forced
from causal_agent.families.adjustment.lane.contracts import Contrasts, DesignAssessment, EstimatorPick, Mechanism, PostRole, PostRoles, Revision, Role, Roles
from causal_agent.families.adjustment.lane.graph import compile_local
from causal_agent.memory import store

CITE = "col:test_preparation_course.note"


def _memory(name):
    """The memory from the shipped claims file alone: a note mined into the memory on disk never moves a test."""
    return store.migrate(name, write=False)


def students_handoff() -> Handoff:
    cols = ["math score", "test preparation course", "lunch", "parental level of education", "gender", "race/ethnicity", "reading score", "writing score"]
    return forced(
        "students",
        "Did completing the prep course raise math scores?",
        "adjustment",
        "math score",
        "test preparation course",
        cols,
        assumption="nothing beyond lunch and parental education drove both",
        cite=CITE,
        memory=_memory("students"),
    )


def uruguay_handoff() -> Handoff:
    cols = ["Support", "Participation", "Income_Centered", "Education", "Age"]
    return forced(
        "gov_transfers",
        "Did receiving the transfer raise support for the government?",
        "adjustment",
        "Support",
        "Participation",
        cols,
        assumption="forced into the adjustment lane for the negative case",
        cite="col:participation.note",
        memory=_memory("gov_transfers"),
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
    """Scripted answers per schema. `overrides` sets a column's claims (the four, plus `nested_in`, `redundant_with`, `links`);
    `looks` scripts the tool calls of an episode by node name, one list per round of looking."""

    def __init__(self, *, bad_cites=False, assess_script=None, pick_script=None, interpret_bad_first=False, cite=CITE, overrides=None, looks=None):
        self.bad_cites = bad_cites
        self.assess_script = list(assess_script or [])
        self.pick_script = list(pick_script or [])
        self.interpret_bad_first = interpret_bad_first
        self.cite = cite
        self.overrides = dict(overrides or {})
        self.looks = {k: list(v) for k, v in (looks or {}).items()}
        self.calls: list[str] = []
        self.humans: list[tuple[str, str]] = []

    def humans_of(self, name: str) -> list[str]:
        return [h for n, h in self.humans if n == name]

    def asked(self, name: str) -> list[str]:
        """The columns a rung's prompt asked to place, across every call."""
        return [c for h in self.humans_of(name) for c in re.findall(r"^COLUMN '([^']+)'", h, re.M)]

    # the tool phase of an episode
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        m = re.search(r"\[probe:([a-z_]+)\.<n>\]", messages[0].content)
        node = m.group(1) if m else ""
        rounds = self.looks.get(node) or []
        calls = rounds.pop(0) if rounds else []
        return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"{node}{i}"} for i, (n, a) in enumerate(calls)])

    # the answer
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

    def claims_for(self, col: str, block: str) -> dict:
        claims = dict(affects_treatment=False, affects_outcome=False, affected_by_treatment=False, is_outcome_measure=False)
        claims.update(STUDENT_RELATIONS.get(col, {}))
        if "THE LAST READING" in block:  # keep the last reading, as the prompt asks
            reading = block.split("THE LAST READING")[1]
            claims.update({k: v == "true" for k, v in re.findall(r"^\s+(\w+) = (true|false) \[", reading, re.M)})
        claims.update(self.overrides.get(col, {}))
        return claims

    @staticmethod
    def blocks(human: str) -> list[tuple[str, str]]:
        parts = re.split(r"^COLUMN '([^']+)'\n", human.split("THE COLUMNS TO PLACE")[1], flags=re.M)
        return list(zip(parts[1::2], parts[2::2], strict=True))

    def roles_answer(self, human: str) -> Roles:
        cite = "col:nope.note" if self.bad_cites else self.cite
        items = []
        for col, block in self.blocks(human):
            c = self.claims_for(col, block)
            four = {k: c[k] for k in ("affects_treatment", "affects_outcome", "affected_by_treatment", "is_outcome_measure")}
            reasons = [Cited(reason=f"{col}: {k}", cites=[c.get("cite", cite)]) for k, v in four.items() if v]
            links = [Cited(reason=f"{col}: link", cites=list(c["links"]))] if c.get("links") else []
            items.append(Role(column=col, reasons=reasons, nested_in=c.get("nested_in"), redundant_with=c.get("redundant_with"), links=links, **four))
        return Roles(items=items)

    def post_roles_answer(self, human: str) -> PostRoles:
        cite = "col:nope.note" if self.bad_cites else self.cite
        items = []
        for col, block in self.blocks(human):
            c = self.claims_for(col, block)
            kind = (
                "outcome_measure"
                if c["is_outcome_measure"]
                else "mediator"
                if c["affected_by_treatment"] and c["affects_outcome"]
                else "consequence_of_treatment"
                if c["affected_by_treatment"]
                else "background"
                if c["affects_outcome"]
                else "unrelated"
            )
            items.append(PostRole(column=col, kind=kind, reason=f"{col}: {kind}", cites=[cite]))
        return PostRoles(items=items)

    def answer(self, schema, human):
        self.calls.append(schema.__name__)
        self.humans.append((schema.__name__, human))
        cite = "col:nope.note" if self.bad_cites else self.cite
        if schema is Contrasts:
            levels = re.findall(r"'([^']+)'", human.split("OBSERVED LEVELS")[1].split("\n")[1])
            control = "none" if "none" in levels else levels[0]
            treated = next(v for v in levels if v != control)
            return Contrasts(items=[Contrast(control=control, treated=treated, reason="the note says completed is the course", cites=[cite])])
        if schema is Mechanism:
            return Mechanism(drivers=[], self_selection=True, reason="the story says units chose after an offer", cites=[cite])
        if schema is Roles:
            return self.roles_answer(human)
        if schema is PostRoles:
            return self.post_roles_answer(human)
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
            return Interpretation(
                contrast=contrast,
                answer=f"The effect is {value:.3g} points.",
                effect_stated=value,
                caveats=["bets on the router's assumption"] + [f"flag {a}" for a in required],
                cites=cites,
            )
        raise AssertionError(schema)


@pytest.fixture(autouse=True)
def _restore(tmp_path, monkeypatch):
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
    # students ships no claims: every column's timing is unknown, so one roles episode places all six together; the pair and the
    # mechanism are episodes too, since the pack names neither the treated level nor what the offer looked at
    assert fake.calls.count("Roles") == 1 and fake.calls.count("PostRoles") == 0 and fake.calls.count("Contrasts") == 1 and fake.calls.count("Mechanism") == 1
    assert len(fake.asked("Roles")) == 6 and set(fake.asked("Roles")) <= set(STUDENT_RELATIONS) and fake.calls.count("EstimatorPick") == 1
    assert "DesignAssessment" in fake.calls  # students has a soft balance flag, so assess ran
    nodes = {t.node for t in out["debug"]}
    assert {"pair", "mechanism", "roles", "assess", "pick_estimator", "interpret:completed_vs_none"} <= nodes
    for name in ("Roles", "DesignAssessment", "EstimatorPick", "Interpretation"):  # every judgement sees the case
        assert fake.humans_of(name) and all("THE CASE" in p and "[change:1.note]" in p for p in fake.humans_of(name)), name
    # every rung reads the rungs below it, and the report and the material carry the ladder
    assert "THE LADDER SO FAR" in fake.humans_of("Roles")[0] and "[ladder:mechanism.kind]" in fake.humans_of("Roles")[0]
    assert "[ladder:pair.contrasts]" in fake.humans_of("Mechanism")[0] and "THE LADDER SO FAR" not in fake.humans_of("Contrasts")[0]
    lad = out["ladder"]
    assert lad.pair.by == "judgement" and lad.mechanism.by == "judgement" and lad.timing.unknown == fake.asked("Roles")
    assert {a for a, _ in lad.lines()} >= {"ladder:pair.contrasts", "ladder:mechanism.drivers", "ladder:timing.unknown", "ladder:roles.lunch"}
    assert "THE LADDER" in r["report"] and "[ladder:roles.lunch] confounder" in r["report"] and ["ladder:roles.lunch", "confounder"] in r["ladder"]
    assert (
        "[ladder:roles.lunch] confounder" in fake.humans_of("Interpretation")[0]
        and "ladder:roles.lunch" in fake.humans_of("Interpretation")[0].split("ADDRESSES YOU MAY CITE")[1]
    )
    assert (json.loads(open(f"{r['run_dir']}/design.json").read())["estimator"]) == "propensity_score_stratification"


def test_bad_cites_are_refused_three_times_then_the_rung_stops():
    fake = FakeLLM(bad_cites=True)
    out = _run(fake, _students3())  # the pair and the mechanism are the pack's; the roles rung is the first judgement
    r = out["specialist_result"]
    assert r["status"] == "infeasible"
    assert out["feasibility"].stage == "roles" and fake.calls.count("Roles") == 3 and fake.calls.count("PostRoles") == 0
    assert "col:nope.note is not an address you may cite" in fake.humans_of("Roles")[1] and out["episodes"]["roles"].tries == 3
    assert out.get("design") is None and not out.get("estimates")


def test_revision_is_a_delta_and_relate_not_rerun():
    script = [
        DesignAssessment(
            action="revise",
            reason="lunch is imbalanced; exclude it",
            cites=[],
            revisions=[Revision(column="parental_level_of_education", change="exclude", reason="test delta", cites=[CITE])],
        ),
        DesignAssessment(action="proceed", reason="fine now", cites=[]),
    ]
    fake = FakeLLM(assess_script=script)
    out = _run(fake, students_handoff())
    assert out["specialist_result"]["status"] == "done"
    assert out["revisions"] == 1
    assert "parental_level_of_education" in {x.column for x in out["graph"].excluded}
    assert out["design"].estimand.adjustment_set == ["lunch"]
    assert fake.calls.count("Roles") == 1  # the delta was applied by merge_graph; no rung ran again


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

    from causal_agent.families.adjustment.lane.knowledge import load_checks, load_estimators, load_refuters

    for e in load_estimators():
        assert causal_estimators.get_class_object(e.dowhy.split(".", 1)[1] + "_estimator") is not None, e.name
    for r in load_refuters():
        assert causal_refuters.get_class_object(r.name) is not None, r.name
    cfg = load_checks()
    assert cfg["overlap"]["common_support_share"]["hard"] < cfg["overlap"]["common_support_share"]["soft"]
    assert cfg["balance"]["smd"]["soft"] < cfg["balance"]["smd"]["hard"]


def test_the_desk_reaches_the_real_specialist():
    from causal_agent.families.registry import lanes

    SPECIALISTS = lanes()

    assert {"pair", "mechanism", "time", "roles", "post_roles", "freeze_design"} <= set(SPECIALISTS["adjustment"].get_graph().nodes)
    assert "relate" not in SPECIALISTS["synthetic_control"].get_graph().nodes  # still a stub


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
    from causal_agent.families.adjustment.lane import nodes as N

    material = N._material(out, c.key)  # the beliefs are in what the interpretation reads, with addresses it may cite
    assert "[claim:unobserved] nothing outside the file" in material and "claim:unobserved" in N._addresses(out, c.key)
    # parental education: the offer looked at it and it was fixed before, so its relation is a fact; lunch was marked 'after', so the
    # post-treatment rung places it, with the pack's word that the offer looked at it copied in
    assert "parental_level_of_education" not in fake.asked("Roles") + fake.asked("PostRoles") and "lunch" in fake.asked("PostRoles")
    assert sorted(fake.asked("Roles")) == ["gender", "race_ethnicity"] and sorted(fake.asked("PostRoles")) == ["lunch", "reading_score", "writing_score"]
    assert out["ladder"].mechanism.by == "pack" and out["ladder"].mechanism.drivers == ["lunch", "parental_level_of_education"]
    assert any(e.src == "lunch" and e.dst == "test_preparation_course" for e in out["graph"].edges)
    parental = next(e for e in out["graph"].edges if e.src == "parental_level_of_education" and e.dst == "test_preparation_course")
    assert "claim:assignment.depends_on" in parental.cites
    # the person's words reach every judgement the lane makes (students3 ships no transcript, so the section is there and empty)
    from causal_agent.lane import nodes as L

    assert "WHAT THE PERSON SAID" in L.frame_text(out)


# ------------------------------------------------------------------ the other roads: a hidden factor, an instrument, a mediator


def _synthetic(tmp_path, seed=0, groups=False):
    """z pushes units into treatment and touches y no other way; m carries the whole effect (y = 2m + u); u drives both t and y.
    With `groups`, g is a fine group and g2 the coarser group every g sits inside."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = 600
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    t = (z + u + rng.normal(size=n) > 0).astype(int)
    m = t + rng.normal(size=n)
    y = 2 * m + u + rng.normal(size=n)
    csv = tmp_path / "synthetic.csv"
    frame = pd.DataFrame({"z": z, "treated": t, "m": m, "y": y})
    if groups:
        frame["g"] = [f"g{v}" for v in rng.integers(0, 6, n)]
        frame["g2"] = frame["g"].map(lambda v: "north" if int(v[1]) < 3 else "south")
    frame.to_csv(csv, index=False)
    return csv


def _synthetic_memory(csv, *, hidden=True, instrument=None, mediator=None, said_none=False, groups=False):
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
    for c, when in (("y", "after"), ("treated", "at"), ("z", "before"), ("m", "after")) + ((("g", "before"), ("g2", "before")) if groups else ()):
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


def _synthetic_handoff(memory, columns=("y", "treated", "z", "m")):
    return forced(
        "synthetic",
        "Did the programme raise y?",
        "adjustment",
        "y",
        "treated",
        list(columns),
        assumption="the instrument and the mediator are as the person says",
        cite="col:z.note",
        memory=memory,
    )


def test_instrument_and_mediator_open_roads_around_a_hidden_factor(tmp_path):
    csv = _synthetic(tmp_path)
    h = _synthetic_handoff(_synthetic_memory(csv, hidden=True, instrument="z", mediator="m"))
    assert h.design.instrument == "z" and h.design.mediator == "m" and h.design.unobserved_confounding is True
    fake = FakeLLM(pick_script=["instrumental_variable"], cite="col:z.note")
    out = _run(fake, h, question="Did the programme raise y?")
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    assert fake.calls.count("Roles") == 0 and fake.calls.count("PostRoles") == 0  # the instrument and the mediator are the person's word: facts, not judgements
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


def _students3(cols=None, scope=None, memory=None):
    from causal_agent.common.contracts import Scope

    cols = cols or [
        "math score",
        "test preparation course",
        "lunch",
        "parental level of education",
        "gender",
        "race/ethnicity",
        "reading score",
        "writing score",
    ]
    return forced(
        "students3",
        "Did completing the prep course raise math scores?",
        "adjustment",
        "math score",
        "test preparation course",
        cols,
        cite=CITE,
        scope=scope or Scope(),
        memory=memory or _memory("students3"),
    )


def test_a_forbidden_column_never_enters_the_graph_and_the_flags_are_cited():
    """students3 marks the two other scores 'after': the pack forbids them. A model that calls one of them a plain parent of the
    outcome is overruled by code; the other it calls a measure of the outcome, which is excluded first."""
    h = _students3()
    assert {"reading_score", "writing_score"} <= set(h.design.forbidden)

    fake = FakeLLM(overrides={"writing_score": dict(affects_outcome=True, is_outcome_measure=False)})
    out = _run(fake, h)
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    why = {x.column: x.why for x in out["graph"].excluded}
    assert "design.forbidden" in why["writing_score"] and "measurement of the outcome" in why["reading_score"]
    assert not any(e.src in ("reading_score", "writing_score") for e in out["graph"].edges)
    # the case reached the roles prompt: the settled block for a column the pack partly settles, the pack's cards and probes
    human = fake.humans_of("Roles")[0]
    assert (
        "COLUMN 'gender'" in human
        and "SETTLED BY THE PACK" in human
        and "affected_by_treatment = false [col:gender.when]" in human
        and "[probe:adjustment" in human
        and "[dataset.profile.rows]" in human
    )
    # the interpretation had to cite every flag, and the artifacts carry the case and the checks
    assert not out.get("interpret_errors")
    arts = json.loads(open(f"{r['run_dir']}/artifacts.json").read())
    assert arts["case"]["facts"]["col:gender.when"] == "before" and any(c["name"] == "arms" for c in arts["checks"]) and arts["design"]["estimator"]
    ids = [f["id"] for f in json.loads(open(f"{r['run_dir']}/figures.json").read())]
    assert "effect_completed_vs_none" in ids and r["figures"] == ids


def test_a_column_the_pack_settles_is_never_placed_by_a_judgement():
    """A column the offer looked at and that was fixed before is a fact the graph takes; a column the pack only partly settles is
    placed by its rung, with the settled claims shown and copied."""
    h = _students3()
    fake = FakeLLM()
    out = _run(fake, h)
    asked = set(fake.asked("Roles"))
    assert "parental_level_of_education" not in asked and "gender" in asked
    from causal_agent.families.adjustment.lane import nodes as N

    claims, cites = N.settled_claims(h, "gender", out["case"])
    assert claims == {"affected_by_treatment": False} and cites == {"affected_by_treatment": "col:gender.when"}
    claims, _ = N.settled_claims(h, "lunch", out["case"])
    assert claims.get("affects_treatment") is True  # the offer depended on it, whatever the timing says


def test_a_confirmed_relation_settles_the_claim_without_a_model_call_and_a_drafted_one_is_the_last_reading():
    """The relationships are memory claims: gender's four, confirmed by the person, make its relation a fact the lane takes with no
    judgement; race's `feeds_treatment`, drafted by an earlier run, is shown to the model as the last reading."""
    from causal_agent.families.adjustment.lane import nodes as N

    m = _memory("students3")
    for field, value in (("feeds_treatment", False), ("moves_outcome", True), ("measures_outcome", False)):
        m.set(f"col:gender.{field}", value, status="confirmed", source="user:turn:5", said="gender never fed the offer but marks differ by it")
    m.set("col:race_ethnicity.feeds_treatment", True, status="drafted", source="model:relate", reason="the run's graph drew this edge")
    h = _students3(memory=m)
    assert "[col:gender.feeds_treatment] did not feed the decision or the offer · confirmed · user:turn:5" in h.brief_text("gender")
    fake = FakeLLM()
    out = _run(fake, h)
    assert out["specialist_result"]["status"] == "done", out["specialist_result"].get("feasibility")
    asked = set(fake.asked("Roles"))
    assert "gender" not in asked and "race_ethnicity" in asked
    claims, cites = N.settled_claims(h, "gender", out["case"])
    assert claims == {"affects_treatment": False, "affects_outcome": True, "affected_by_treatment": False, "is_outcome_measure": False}
    assert cites["affects_treatment"] == "col:gender.feeds_treatment" and cites["affects_outcome"] == "col:gender.moves_outcome"
    assert any(e.src == "gender" and e.dst == "math_score" for e in out["graph"].edges) and not any(
        e.src == "gender" and e.dst == "test_preparation_course" for e in out["graph"].edges
    )
    human = fake.humans_of("Roles")[0]
    race = human.split("COLUMN 'race_ethnicity'")[1].split("COLUMN '")[0]
    assert (
        "THE LAST READING (drafted; depart from it only with a cited reason)" in race
        and "affects_treatment = true [col:race_ethnicity.feeds_treatment]" in race
    )
    assert N.drafted_claims(h, "race_ethnicity", out["case"]) == ({"affects_treatment": True}, {"affects_treatment": "col:race_ethnicity.feeds_treatment"})
    assert "THE LAST READING" not in fake.humans_of("PostRoles")[0].split("COLUMN 'lunch'")[1].split("COLUMN '")[0]
    assert any(e.src == "race_ethnicity" and e.dst == "test_preparation_course" for e in out["graph"].edges)  # the reading was kept


def test_a_departure_from_the_last_reading_with_no_departure_named_is_refused_once():
    m = _memory("students3")
    m.set("col:race_ethnicity.feeds_treatment", True, status="drafted", source="model:relate", reason="the run's graph drew this edge")
    h = _students3(memory=m)

    class Fake(FakeLLM):
        def roles_answer(self, human):
            out = super().roles_answer(human)
            if len(self.humans_of("Roles")) == 1:  # the first answer departs from the drafted reading and names no departure
                for x in out.items:
                    if x.column == "race_ethnicity":
                        x.affects_treatment = False
                        x.reasons = [r for r in x.reasons if not r.reason.endswith("affects_treatment")]
            return out

    fake = Fake()
    out = _run(fake, h)
    assert out["specialist_result"]["status"] == "done"
    seen = fake.humans_of("Roles")
    assert len(seen) == 2 and "race_ethnicity: affects_treatment = False departs from the last reading True with no departure named" in seen[1]
    assert out["episodes"]["roles"].tries == 2
    assert any(e.src == "race_ethnicity" and e.dst == "test_preparation_course" for e in out["graph"].edges)  # the second answer kept the reading


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
    assert (
        r["status"] == "ask" and r["ask"]["address"] == "claim:mediator.column" and "Which column" in r["ask"]["question"] and r["ask"]["stage"] == "identify"
    )


def test_sensitivity_survives_a_revise_loop_and_a_repick_does_not_duplicate_estimates(tmp_path):
    csv = _synthetic(tmp_path)
    h = _synthetic_handoff(_synthetic_memory(csv, hidden=True, said_none=True))
    # z read as a parent of both puts it in the adjustment set; its imbalance is flagged; the assessment revises it out, then proceeds
    script = [
        DesignAssessment(
            action="revise",
            reason="z is too imbalanced to adjust for",
            cites=[],
            revisions=[Revision(column="z", change="exclude", reason="test delta", cites=["col:z.note"])],
        ),
        DesignAssessment(action="proceed", reason="fine", cites=[]),
    ]

    fake = FakeLLM(
        overrides={"z": dict(affects_treatment=True, affects_outcome=True)},
        assess_script=script,
        cite="col:z.note",
        pick_script=["econml_magic", "linear_regression"],
    )
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
    assert (
        roles["test_preparation_course"] == "treatment"
        and roles["math_score"] == "outcome"
        and roles["parental_level_of_education"] == "confounder"
        and roles["reading_score"] == "excluded"
    )
    assert any(
        e["src"] == "parental_level_of_education" and e["dst"] == "test_preparation_course" and "claim:assignment.depends_on" in e["cites"] for e in g["edges"]
    )
    b = figs[1]
    assert [s["name"] for s in b["series"]] == ["before adjustment", "after weighting on the score"] and set(b["series"][0]["x"]) == {
        "lunch",
        "parental level of education",
    }
    assert all(a is not None and a < 0.3 for a in b["series"][1]["y"])
    assert not any(x["check"] == "figure.check" for x in r["declines"])


# ------------------------------------------------------------------ the design brief names the road


def _with_road(h: Handoff, road: str) -> Handoff:
    from causal_agent.common.contracts import DecisionMade, DesignBrief

    h.brief = DesignBrief(
        family="adjustment",
        road=road,
        decisions=[DecisionMade(name="road", choice=f"the {road} road", rests_on=["claim:assignment.kind"], reason="scripted")],
        bets_on="the brief's own sentence",
    )
    h.chosen_assumption = h.brief.bets_on
    return h


def test_a_brief_naming_a_road_the_graph_does_not_open_stops_at_identify():
    """students has no instrument: a brief that names the iv road is an honest stop, with the roads found as the facts."""
    fake = FakeLLM()
    out = _run(fake, _with_road(students_handoff(), "iv"))
    r = out["specialist_result"]
    assert r["status"] == "infeasible" and out["feasibility"].stage == "identify"
    assert out["feasibility"].reason == "the design brief names the iv road and the graph has no such road"
    assert out["feasibility"].facts[0] == "roads found: backdoor" and out.get("design") is None and fake.calls.count("EstimatorPick") == 0
    assert "DESIGN BRIEF" in fake.humans_of("Roles")[0] and "[design.brief.road] iv: the iv road" in fake.humans_of("Roles")[0]


def test_a_brief_naming_the_back_door_keeps_it_and_the_pick_sees_it():
    fake = FakeLLM()
    out = _run(fake, _with_road(students_handoff(), "backdoor"))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    d = out["design"]
    assert d.estimand.kind == "backdoor" and set(d.estimand.adjustment_set) == {"lunch", "parental_level_of_education"}
    assert '"road_from_brief": "backdoor"' in fake.humans_of("EstimatorPick")[0]
    from causal_agent.families.adjustment.lane import nodes as N

    c = out["contrasts"][0]
    assert "[design.brief.bets_on] the brief's own sentence" in N._material(out, c.key) and "design.brief.bets_on" in N._addresses(out, c.key)


def test_a_brief_naming_the_front_door_takes_it_over_the_instrument(tmp_path):
    """Both roads are open around the hidden factor; the brief says which the design takes, so the catalogue offers only that road."""
    csv = _synthetic(tmp_path)
    h = _with_road(_synthetic_handoff(_synthetic_memory(csv, hidden=True, instrument="z", mediator="m")), "frontdoor")
    fake = FakeLLM(cite="col:z.note")
    out = _run(fake, h, question="Did the programme raise y?")
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    assert {"iv", "frontdoor"} <= set(out["estimand"].roads) and out["estimand"].kind == "frontdoor"
    assert out["design"].estimator == "frontdoor_two_stage" and out["design"].estimand.kind == "frontdoor"
    assert "NAMES YOU MAY PICK: frontdoor_two_stage" in fake.humans_of("EstimatorPick")[0]


# ------------------------------------------------------------------ the episodes: looking at the data, and the outcome rule


def test_a_rung_may_look_at_the_data_and_no_look_joins_the_outcome_with_the_treatment():
    """The roles rung asks for the outcome by arm and for gender by arm. The first is refused, by code, and the refusal is what the
    model reads; the second becomes a fact with an address the answer cites, and the report, the material and the result carry it."""
    looks = {"roles": [[("by_arm", {"column": "math_score"}), ("by_arm", {"column": "gender"})], []]}
    fake = FakeLLM(looks=looks, overrides={"gender": dict(cite="probe:roles.1")})
    out = _run(fake, _students3())
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    log = out["episodes"]["roles"]
    assert (
        log.calls == 2 and [f.address for f in log.facts] == ["probe:roles.1"] and log.facts[0].tool == "by_arm" and log.facts[0].args == {"column": "gender"}
    )
    assert (
        len(log.refusals) == 1
        and log.refusals[0].args == {"column": "math_score"}
        and "join the outcome 'math_score' with the treatment" in log.refusals[0].reason
    )
    human = fake.humans_of("Roles")[0]
    assert "FACTS YOU ASKED FOR\n[probe:roles.1] by_arm(column='gender'): 'gender' by arm" in human and "'math_score' by arm" not in human
    gender = next(e for e in out["graph"].edges if e.src == "gender" and e.dst == "math_score")
    assert gender.cites == ["probe:roles.1"]
    assert "WHAT THE EPISODES LOOKED AT" in r["report"] and "refused by_arm(column='math_score')" in r["report"] and "[probe:roles.1]" in r["report"]
    assert r["facts"][0]["address"] == "probe:roles.1" and r["facts"][0]["value"] is not None
    from causal_agent.families.adjustment.lane import nodes as N

    c = out["contrasts"][0]
    assert "[probe:roles.1] by_arm(column='gender')" in N._material(out, c.key) and "probe:roles.1" in N._addresses(out, c.key)


def test_two_nested_columns_are_read_together_and_the_graph_keeps_the_finer_one(tmp_path):
    """g sits inside g2. The pack's data facts already say so; the roles rung reads both as confounders and names the nesting on
    that fact, and the graph keeps g and drops g2 with the reason. A nesting claimed on no fact is refused."""
    csv = _synthetic(tmp_path, groups=True)
    h = _synthetic_handoff(_synthetic_memory(csv, hidden=False, groups=True), columns=("y", "treated", "z", "m", "g", "g2"))
    fact = next(p for p in h.probes if p.name == "redundancy.g~g2")
    assert "'g' sits inside 'g2'" in fact.detail
    both = dict(affects_treatment=True, affects_outcome=True)
    fake = FakeLLM(cite="col:z.note", overrides={"g": {**both, "nested_in": "g2", "links": [fact.address]}, "g2": both})
    out = _run(fake, h, question="Did the programme raise y?")
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    g = out["graph"]
    assert "g" in g.nodes and "g2" not in g.nodes
    why = {x.column: x.why for x in g.excluded}
    assert "g sits inside it and is kept" in why["g2"] and "[ladder:roles.g]" in why["g2"]
    assert "g" in out["design"].estimand.adjustment_set and "g2" not in out["design"].estimand.adjustment_set
    assert "[ladder:roles.g] confounder; sits inside g2" in r["report"]
    # the same claim on a cite that is not a redundancy fact is refused, three times, and the rung stops
    fake = FakeLLM(cite="col:z.note", overrides={"g": {**both, "nested_in": "g2", "links": ["col:z.note"]}, "g2": both})
    out = _run(fake, h, question="Did the programme raise y?")
    assert out["specialist_result"]["status"] == "infeasible" and out["feasibility"].stage == "roles"
    assert "g: nested_in = 'g2' needs a fact under links" in fake.humans_of("Roles")[1]


def test_a_look_that_finds_the_nesting_backs_the_claim_too(tmp_path):
    """No pack fact this time: the rung asks the redundancy tool, and the tool's answer is the fact the claim rests on."""
    csv = _synthetic(tmp_path, groups=True)
    h = _synthetic_handoff(_synthetic_memory(csv, hidden=False, groups=True), columns=("y", "treated", "z", "m", "g", "g2"))
    h.probes = [p for p in h.probes if not p.name.startswith("redundancy.")]
    both = dict(affects_treatment=True, affects_outcome=True)
    fake = FakeLLM(
        cite="col:z.note",
        looks={"roles": [[("redundancy", {"a": "g", "b": "g2"})], []]},
        overrides={"g": {**both, "nested_in": "g2", "links": ["probe:roles.1"]}, "g2": both},
    )
    out = _run(fake, h, question="Did the programme raise y?")
    assert out["specialist_result"]["status"] == "done", out["specialist_result"].get("feasibility")
    assert out["episodes"]["roles"].facts[0].tool == "redundancy" and "'g' sits inside 'g2'" in out["episodes"]["roles"].facts[0].text
    assert "g2" not in out["graph"].nodes and "g" in out["design"].estimand.adjustment_set
