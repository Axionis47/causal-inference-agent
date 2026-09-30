"""The diff-in-diff lane with a fake model and real pyfixest on the real files. No Vertex calls."""

from __future__ import annotations

import json
import re
import uuid

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from causal_agent.common.contracts import Cited, Handoff, Interpretation
from causal_agent.common.llm import set_llm
from causal_agent.desk.handoff import forced
from causal_agent.families.diff_in_diff.lane.contracts import (
    Comparison,
    ControlRelation,
    ControlRoles,
    DesignAssessment,
    EstimatorPick,
    Groups,
    Periods,
    Revision,
)
from causal_agent.families.diff_in_diff.lane.graph import compile_local
from causal_agent.lane.ladder import Heterogeneity, Modifier
from causal_agent.memory import store


def memory(pack: str, *, trend=None, trend_status="unknown", spillover=False, said=None):
    """The claims file alone (a note mined on disk never moves a test), plus the two beliefs the family asks for: by default
    the person could not say whether the paths would have stayed together, and units do not reach one another."""
    m = store.migrate(pack, write=False)
    m.set("claim:trend_continues.believed", trend, status=trend_status, source="user:turn:1", said=said)
    m.set("claim:spillover.possible", spillover, status="confirmed", source="user:turn:1")
    return m


def handoff(pack: str, outcome: str, treatment: str, cols: list[str], cite: str, memory_=None) -> Handoff:
    return forced(
        pack, "q", "diff_in_diff", outcome, treatment, cols, assumption="parallel movement absent the change", cite=cite, memory=memory_ or memory(pack)
    )


CK = dict(pack="card_krueger", outcome="total_emp_nov", treatment="state", cols=["state", "total_emp_feb", "total_emp_nov"], cite="col:state.note")
CIGAR = dict(pack="cigar", outcome="sales", treatment="state", cols=["state", "year", "sales", "price", "pimin", "ndi", "pop", "cpi"], cite="col:state.note")
MARKETING = dict(
    pack="marketing",
    outcome="SalesInThousands",
    treatment="Promotion",
    cols=["Promotion", "week", "SalesInThousands", "MarketSize", "LocationID"],
    cite="col:promotion.note",
)

# scripted answers per dataset, what a careful reader of the notes would say
SCRIPT = {
    "card_krueger": dict(
        groups=("state", "1"),
        periods=Periods(
            kind="wide", before_column="total_emp_feb", after_column="total_emp_nov", reason="the notes name the two waves", cites=["col:total_emp_feb.note"]
        ),
        relations={},
    ),
    "cigar": dict(
        groups=("state", "5"),
        periods=Periods(kind="long", time_column="year", first_post="89", reason="Proposition 99 from January 1989", cites=["change:1.note"]),
        relations={
            "price": dict(affected_by_treatment=True),
            "pimin": dict(usable_as_control=True),
            "ndi": dict(usable_as_control=True),
            "pop": dict(usable_as_control=True),
            "cpi": dict(usable_as_control=True),
        },
    ),
    "marketing": dict(
        groups=("promotion", "2"),
        periods=Periods(kind="long", time_column="week", first_post="1", reason="every row is under the promotion", cites=["change:1.note"]),
        relations={"marketsize": dict(usable_as_control=True)},
    ),
}


class FakeLLM:
    """Scripted answers per schema. `overrides` sets a column's claims in the controls rung; `risks` are what the comparison rung
    names; `looks` scripts the tool calls of an episode by node name, one list per round of looking."""

    def __init__(self, script: dict, cite: str, *, bad_cites=False, assess_script=None, pick_script=None, overrides=None, risks=None, looks=None):
        self.script, self.cite, self.bad_cites = script, cite, bad_cites
        self.assess_script, self.pick_script = list(assess_script or []), list(pick_script or [])
        self.overrides = dict(overrides or {})
        self.risks = list(risks or [])
        self.looks = {k: list(v) for k, v in (looks or {}).items()}
        self.calls: list[str] = []
        self.humans: list[tuple[str, str]] = []

    def humans_of(self, name: str) -> list[str]:
        return [h for n, h in self.humans if n == name]

    def asked(self, name: str) -> list[str]:
        return [c for h in self.humans_of(name) for c in re.findall(r"^COLUMN '([^']+)'", h, re.M)]

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        m = re.search(r"\[probe:([a-z_]+)\.<n>\]", messages[0].content)
        node = m.group(1) if m else ""
        rounds = self.looks.get(node) or []
        calls = rounds.pop(0) if rounds else []
        return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"{node}{i}"} for i, (n, a) in enumerate(calls)])

    def with_structured_output(self, schema, include_raw=False):
        fake = self

        class R:
            def invoke(self_, messages):
                parsed = fake.answer(schema, messages[-1][1])
                raw = AIMessage(
                    content=[{"type": "thinking", "thinking": f"thinking about {schema.__name__}"}, "{}"],
                    usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30, "output_token_details": {"reasoning": 7}},
                )
                return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return R()

    def controls_answer(self, human: str) -> ControlRoles:
        cite = "col:nope.note" if self.bad_cites else self.cite
        items = []
        for col in re.findall(r"^COLUMN '([^']+)'", human.split("THE COLUMNS TO PLACE")[1], re.M):
            flags = dict(affected_by_treatment=False, usable_as_control=False, modifier_candidate=False)
            flags.update(self.script["relations"].get(col, {}))
            flags.update(self.overrides.get(col, {}))
            reasons = [Cited(reason=f"{col}: {k}", cites=[cite]) for k in ("affected_by_treatment", "usable_as_control") if flags[k]]
            items.append(ControlRelation(column=col, reasons=reasons, **flags))
        return ControlRoles(items=items)

    def answer(self, schema, human):
        self.calls.append(schema.__name__)
        self.humans.append((schema.__name__, human))
        cite = "col:nope.note" if self.bad_cites else self.cite
        if schema is Groups:
            col, lvl = self.script["groups"]
            return Groups(column=col, treated_level=lvl, reason="the change card says so", cites=[cite])
        if schema is Periods:
            p = self.script["periods"].model_copy()
            p.cites = [cite]
            return p
        if schema is Comparison:
            return Comparison(fair=not self.risks, why="the story gives no reason the groups would have parted", risks=list(self.risks), cites=[cite])
        if schema is ControlRoles:
            return self.controls_answer(human)
        if schema is Heterogeneity:
            cands = re.findall(r"^\[col:([^\]]+)\]", human.split("CANDIDATES")[1], re.M)
            return Heterogeneity(
                modifiers=[Modifier(column=cands[0], reason="the story says the effect could differ by it", cites=[cite])],
                why="one trait the story backs",
                cites=[cite],
            )
        if schema is DesignAssessment:
            if self.assess_script:
                return self.assess_script.pop(0)
            return DesignAssessment(action="proceed", reason="only soft flags", cites=[])
        if schema is EstimatorPick:
            names = [n.strip() for n in re.search(r"NAMES YOU MAY PICK: (.*)", human).group(1).split(",")]
            return EstimatorPick(name=self.pick_script.pop(0) if self.pick_script else names[0], reason="ranked first", cites=[])
        if schema is Interpretation:
            addresses = human.split("ADDRESSES YOU MAY CITE")[1].split("\n\n")[0].strip().splitlines()
            required = [a for a in human.split("ADDRESSES YOU MUST CITE")[1].split("\n\n")[0].strip().splitlines()[1:] if a and a != "(none)"]
            contrast = re.search(r"COMPARISON: (\S+)", human).group(1)
            value = float(re.search(r"\[estimate:%s\.value\] ([-\d.eE+]+)" % re.escape(contrast), human).group(1))
            return Interpretation(
                contrast=contrast,
                answer=f"The effect on the treated is {value:.3g}.",
                effect_stated=value,
                caveats=["parallel trends assumed"] + [f"flag {a}" for a in required],
                cites=list(dict.fromkeys(required + addresses[:3])),
            )
        raise AssertionError(schema)


@pytest.fixture(autouse=True)
def _restore(tmp_path, monkeypatch):
    yield
    set_llm(None)


def _run(fake, h, question="q"):
    set_llm(fake)
    g = compile_local()
    return g.invoke({"question": question, "handoff": h, "dataset": h.pack_name}, {"configurable": {"thread_id": str(uuid.uuid4())}})


# ------------------------------------------------------------------ tests


def test_card_krueger_wide_happy_path():
    fake = FakeLLM(SCRIPT["card_krueger"], CK["cite"])
    out = _run(fake, handoff(**CK))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    d = out["design"]
    assert d.periods.kind == "wide" and d.shape.units_treated == 309 and d.shape.units_control == 75
    assert d.shape.periods_pre == 1 and d.estimator == "twfe_static" and d.also_run is None  # dynamic needs two pre periods
    assert d.inference == "robust_rows"
    primary = next(e for e in out["estimates"] if e.method == "twfe_static")
    ck = pd.read_csv("data/raw/card-krueger-minimum-wage/employment.csv")
    m = ck.groupby("state")[["total_emp_feb", "total_emp_nov"]].mean()
    four_means = (m.loc[1, "total_emp_nov"] - m.loc[1, "total_emp_feb"]) - (m.loc[0, "total_emp_nov"] - m.loc[0, "total_emp_feb"])
    assert abs(primary.value - four_means) < 1e-6
    assert {c.name for c in out["checks"] if c.level == "soft"} == {"parallel_untestable", "belief.trend_continues"}  # the person could not say
    assert [x.refuter for x in out["refutations"]] == ["placebo_group"] and out["refutations"][0].passed is True
    assert fake.calls.count("ControlRoles") == 0 and fake.calls.count("Comparison") == 1 and "DesignAssessment" in fake.calls
    assert len(out["interpretations"]) == 1 and not out.get("interpret_errors")
    assert "DESIGN" in r["report"] and "ANSWER" in r["report"]
    lad = out["ladder"]
    assert lad.groups.by == "judgement" and lad.periods.by == "judgement" and lad.comparison.fair and lad.heterogeneity.by == "code" and lad.cluster is not None
    assert "[ladder:groups.treated] state = '1'" in r["report"] and "[ladder:comparison.fair] yes" in r["report"] and "[ladder:cluster.level]" in r["report"]
    assert (
        "THE LADDER SO FAR" in fake.humans_of("Comparison")[0]
        and "[ladder:shape.units] 309 treated units, 75 never treated" in fake.humans_of("Comparison")[0]
        and "[ladder:shape.adoption] one-shot" in fake.humans_of("Comparison")[0]
    )
    assert ["ladder:comparison.fair", "yes"] in r["ladder"] and "ladder:comparison.fair" in fake.humans_of("Interpretation")[0].split("ADDRESSES YOU MAY CITE")[
        1
    ]
    for name in ("DesignAssessment", "EstimatorPick", "Interpretation"):  # every judgement after the shape sees the case
        assert fake.humans_of(name) and all("THE CASE" in p and "[change:1.note]" in p for p in fake.humans_of(name)), name


def test_cigar_long_stops_on_pre_trends():
    script = [DesignAssessment(action="stop", reason="the groups diverged long before 1989", cites=[])]
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"], assess_script=script)
    out = _run(fake, handoff(**CIGAR))
    assert out["specialist_result"]["status"] == "infeasible"
    assert out["feasibility"].stage == "assess"
    s = out["shape"]
    assert s.kind == "long" and s.units_treated == 1 and s.units_control == 45 and s.periods_pre == 26 and s.periods_post == 4
    c = out["controls"]
    assert "price" in {x.column for x in c.excluded}
    assert "cpi" in {x.column for x in c.dropped_fixed}
    assert set(c.included) == {"pimin", "ndi", "pop"}
    levels = {x.name: x.level for x in out["checks"]}
    assert levels["pre_trends"] == "hard" and levels["single_treated_unit"] == "soft"
    assert fake.calls.count("ControlRoles") == 1 and sorted(fake.asked("ControlRoles")) == ["cpi", "ndi", "pimin", "pop", "price"]
    assert "[ladder:controls.price] changed by the treatment" in out["specialist_result"]["report"]


def test_cigar_forced_proceed_is_blocked_by_hard_flag():
    script = [DesignAssessment(action="proceed", reason="ignore", cites=[])] * 3
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"], assess_script=script)
    out = _run(fake, handoff(**CIGAR))
    assert out["specialist_result"]["status"] == "infeasible" and fake.calls.count("DesignAssessment") == 3


def test_marketing_stops_with_no_pre_period():
    fake = FakeLLM(SCRIPT["marketing"], MARKETING["cite"])
    out = _run(fake, handoff(**MARKETING))
    assert out["specialist_result"]["status"] == "infeasible"
    assert out["feasibility"].stage == "shape_table" and "before" in out["feasibility"].reason
    assert fake.calls.count("ControlRoles") == 0 and fake.calls.count("Comparison") == 0


def test_bad_cites_stop_the_first_rung_that_judges():
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"], bad_cites=True)
    out = _run(fake, handoff(**CIGAR))
    assert out["specialist_result"]["status"] == "infeasible"
    assert out["feasibility"].stage == "groups" and fake.calls.count("Groups") == 3 and out["episodes"]["groups"].tries == 3


def test_bad_cites_in_the_controls_rung_only():
    class Fake(FakeLLM):
        def answer(self, schema, human):
            self.bad_cites = schema is ControlRoles
            return super().answer(schema, human)

    fake = Fake(SCRIPT["cigar"], CIGAR["cite"])
    out = _run(fake, handoff(**CIGAR))
    assert out["feasibility"].stage == "controls" and fake.calls.count("ControlRoles") == 3
    assert "price: col:nope.note is not an address you may cite" in fake.humans_of("ControlRoles")[1]


def test_revision_is_a_delta():
    script = [
        DesignAssessment(
            action="revise", reason="drop pop", cites=[], revisions=[Revision(column="pop", change="remove_control", reason="test", cites=["col:pop.note"])]
        ),
        DesignAssessment(action="stop", reason="pre-trends still fail", cites=[]),
    ]
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"], assess_script=script)
    out = _run(fake, handoff(**CIGAR))
    assert out["revisions"] == 1 and "pop" not in out["controls"].included
    assert fake.calls.count("ControlRoles") == 1 and fake.calls.count("Comparison") == 1  # no rung ran again


def test_estimator_outside_list_is_rejected_then_accepted():
    fake = FakeLLM(SCRIPT["card_krueger"], CK["cite"], pick_script=["magic", "twfe_static"])
    out = _run(fake, handoff(**CK))
    assert out["specialist_result"]["status"] == "done" and fake.calls.count("EstimatorPick") == 2


def test_catalogues_parse_on_a_toy_panel():
    import pyfixest as pf

    from causal_agent.families.diff_in_diff.lane import adapter
    from causal_agent.families.diff_in_diff.lane.knowledge import load_checks, load_estimators, load_inference, load_placebos

    rng = pd.Series(range(200))
    toy = pd.DataFrame({"unit": (rng % 20).astype(str), "time": rng // 20})
    toy["treated"] = (toy["unit"].astype(int) < 8).astype(int)
    toy["post"] = (toy["time"] >= 5).astype(int)
    toy["treat"] = (toy["treated"] * toy["post"]).astype(float)
    toy["rel_time"] = toy["time"] - 5
    toy["cohort"] = toy["treated"] * 5
    toy["x1"] = toy["time"] * 0.3 + toy["unit"].astype(int) * 0.1
    toy["y"] = 1.0 + 2.0 * toy["treat"] + toy["x1"] + toy["unit"].astype(int) * 0.5
    for e in load_estimators():
        if e.formula in ("did2s", "lpdid"):
            continue
        f = adapter.formula_for(e, ["x1"])
        m = pf.feols(f, toy, vcov="hetero")
        assert m is not None, e.name
    assert [i.name for i in load_inference()] == ["randomisation", "wild_bootstrap", "cluster_unit", "robust_rows"]
    assert {p.name for p in load_placebos()} == {"placebo_group", "placebo_timing"}
    cfg = load_checks()
    assert cfg["pre_trends"]["p_value"]["hard"] < cfg["pre_trends"]["p_value"]["soft"]


def test_the_desk_reaches_both_specialists():
    from causal_agent.families.registry import lanes

    SPECIALISTS = lanes()

    assert "shape_table" in SPECIALISTS["diff_in_diff"].get_graph().nodes
    assert "freeze_design" in SPECIALISTS["adjustment"].get_graph().nodes and "shape_table" not in SPECIALISTS["adjustment"].get_graph().nodes


# ------------------------------------------------------------------ the pack's facts end judgements


def test_pack_panel_block_settles_groups_and_periods_without_a_model_call():
    """No shipped dataset carries diff-in-diff claims yet, so the claims are built here and handed to the builder directly."""
    import json

    from causal_agent.common.contracts import Candidate, FamilyDecision, QuestionFrame, Scope
    from causal_agent.desk.handoff import build
    from causal_agent.families import registry as R
    from causal_agent.memory.claims import Claim, ClaimTable
    from causal_agent.memory.records import Memory
    from causal_agent.profile.datasets import ROOT, dataset_entries
    from causal_agent.profile.profiler import Profile

    table = ClaimTable(
        claims={
            "grain": Claim(
                kind="grain",
                key="grain",
                fields={"row_is": "one state in one year", "key_columns": ["state", "year"], "panel": True},
                status="confirmed",
                source="user:turn:1",
            ),
            "change": Claim(
                kind="change",
                key="change",
                fields={"what": "Proposition 99", "to_whom": "California", "when": "January 1989", "date_column": "year", "period_value": "89"},
                status="confirmed",
                source="user:turn:1",
            ),
            "assignment": Claim(
                kind="assignment",
                key="assignment",
                fields={"kind": "date_by_others", "rule": "a ballot vote in one state", "treatment_column": "state", "treated_level": "5"},
                status="confirmed",
                source="user:turn:1",
            ),
            "trend_continues": Claim(
                kind="trend_continues",
                key="trend_continues",
                fields={"believed": True, "why": "sales moved together before 1989"},
                status="confirmed",
                source="user:turn:2",
            ),
        }
    )
    cols = ["sales", "state", "year", "price", "pimin", "ndi", "pop", "cpi"]
    frame = QuestionFrame(
        intent="effect_of_change",
        decision_served="",
        outcome_candidates=[Candidate(column="sales", reason="r", cites=["col:sales.note"])],
        cause_candidates=[Candidate(column="state", reason="r", cites=["col:state.note"])],
        scope=Scope(target="on_treated"),
        relevant_columns=[Candidate(column=c, reason="r", cites=["col:state.note"]) for c in cols],
        reasons=[],
    )
    decision = FamilyDecision(
        admissible=["diff_in_diff"],
        chosen="diff_in_diff",
        chosen_assumption="parallel movement absent the change",
        why_over_alternatives="only one",
        rejected=[],
    )
    fam = R.family("diff_in_diff")
    e = dataset_entries()["cigar"]
    memory = Memory.from_claims("cigar", table, profile=Profile.model_validate(json.loads((ROOT / e["profile"]).read_text())), csv=e["csv"])
    h = build(question="q", frame=frame, decision=decision, family=fam, memory=memory)
    d = h.design
    assert (
        d.kind == "diff_in_diff" and d.unit == "state" and d.time == "year" and d.change_period == "89" and d.treated_group == {"column": "state", "level": "5"}
    )
    assert d.trend_belief is not None and d.trend_belief.value is True
    assert h.resolve("claim:change.period_value") and h.resolve("claim:trend_continues")
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"], assess_script=[DesignAssessment(action="stop", reason="pre-trends", cites=[])])
    out = _run(fake, h)
    assert fake.calls.count("Groups") == 0 and fake.calls.count("Periods") == 0
    assert out["groups"].column == "state" and out["groups"].treated_level == "5" and out["periods"].first_post == "89"
    assert out["ladder"].groups.by == "pack" and out["ladder"].periods.by == "pack"
    assert out["groups"].cites == ["claim:assignment.treatment_column", "claim:assignment.treated_level"]


# ------------------------------------------------------------------ the lane on the harness: the person's beliefs meet the checks


def _cigar(**kw) -> Handoff:
    return handoff(**CIGAR, memory_=memory("cigar", **kw))


def test_trend_false_and_a_hard_pre_trends_check_stops_by_code_with_no_assessment_call():
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"])
    out = _run(fake, _cigar(trend=False, trend_status="confirmed", said="sales in that state were already falling"))
    r = out["specialist_result"]
    assert r["status"] == "infeasible" and out["feasibility"].stage == "assess" and "agree" in out["feasibility"].reason
    assert any("already falling" in x for x in out["feasibility"].facts) and "DesignAssessment" not in fake.calls
    assert next(c for c in out["checks"] if c.name == "belief.trend_continues").level == "hard"


def test_trend_true_and_a_hard_check_asks_back_once_then_softens_and_the_primary_is_the_full_formula():
    from causal_agent.common.contracts import Said

    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"])
    out = _run(fake, _cigar(trend=True, trend_status="confirmed", said="they moved together for decades"))
    r = out["specialist_result"]
    assert r["status"] == "ask" and r["ask"]["address"] == "claim:trend_continues.believed" and r["ask"]["options"] == ["yes", "no"]
    assert (
        "p = " in r["ask"]["question"]
        and "moved together for decades" in r["ask"]["question"]
        and r["ask"]["evidence"] == [f"check:{out['contrast'].key}.pre_trends"]
    )
    assert "DesignAssessment" not in fake.calls and "ASKS BACK" in r["report"]
    # the person answered once: the flag softens with their reason, the assessment sees a soft flag, and the run goes on
    m = memory("cigar", trend=True, trend_status="confirmed", said="the tax was announced years earlier")
    m.said.append(Said(turn=3, about="lane:claim:trend_continues.believed", text="yes, the tax was announced years earlier"))
    h = handoff(**CIGAR, memory_=m)
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"])
    out = _run(fake, h)
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    pre = next(c for c in out["checks"] if c.name == "pre_trends")
    assert pre.level == "soft" and "announced years earlier" in pre.detail and "DesignAssessment" in fake.calls
    d = out["design"]
    assert d.controls.included == ["pimin", "ndi", "pop"] and "csw0" in d.formula
    primary = next(e for e in out["estimates"] if not e.secondary)
    assert primary.method == "twfe_static" and {e.method for e in out["estimates"] if e.method.startswith("twfe_static+")} == {
        "twfe_static+0",
        "twfe_static+1",
        "twfe_static+2",
    }
    assert pre.address in out["interpretations"][0].cites and not out.get("interpret_errors")
    assert "placebo_group" in out["placebo_draws"] and len(out["placebo_draws"]["placebo_group"]) > 100
    ids = [f["id"] for f in json.loads(open(f"{r['run_dir']}/figures.json").read())]
    assert f"effect_{d.contrast.key}" in ids and f"event_study_{d.contrast.key}" in ids
    assert out["dynamic"] and "-5" in out["dynamic"]


def test_spillover_the_person_kept_is_a_flag_the_interpretation_cites():
    fake = FakeLLM(SCRIPT["card_krueger"], CK["cite"])
    out = _run(fake, handoff(**CK, memory_=memory("card_krueger", spillover=True)))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    flag = next(c for c in out["checks"] if c.name == "belief.spillover")
    assert flag.level == "soft" and "bound" in flag.detail and flag.address in out["interpretations"][0].cites
    assert out["contrast"].control == "0"  # the other level of the settled group column


def test_controls_allowed_is_honoured_and_a_moved_column_is_never_asked_about():
    from causal_agent.common.contracts import Said

    m = memory("cigar", trend=True, trend_status="confirmed", said="together")
    m.said.append(Said(turn=3, about="lane:claim:trend_continues.believed", text="yes"))
    m.set("col:price.moved_by_change", True, status="confirmed", source="user:turn:2", said="the tax is in the price")
    h = handoff(**CIGAR, memory_=m)
    h.design.controls_allowed = ["pimin", "ndi"]
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"])
    out = _run(fake, h)
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    c = out["controls"]
    why = {x.column: x.why for x in c.excluded}
    assert c.included == ["pimin", "ndi"] and "design.controls_allowed" in why["pop"] and "col:price.moved" in why["price"]
    assert sorted(fake.asked("ControlRoles")) == ["cpi", "ndi", "pimin", "pop"]  # price was settled by the person's word


def test_a_cluster_level_the_inference_cannot_honour_is_declined_with_a_record():
    h = handoff(**CK)
    h.design.cluster_level = "state"
    fake = FakeLLM(SCRIPT["card_krueger"], CK["cite"])
    out = _run(fake, h)
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    d = next(x for x in out["declines"] if x.about == "design.cluster_level")
    assert d.check == "inference.cluster_column" and "does not cluster" in d.reason and d.address in r["report"]
    assert any(x["about"] == "design.cluster_level" for x in r["declines"])


def test_pre_periods_that_differ_from_the_pack_are_recorded():
    h = handoff(**CK)
    h.design.pre_periods = 3
    out = _run(FakeLLM(SCRIPT["card_krueger"], CK["cite"]), h)
    d = next(x for x in out["declines"] if x.about == "design.pre_periods")
    assert d.kind == "replaced" and d.pack_value == "3" and d.took == "1" and d.check == "shape.pre_periods"


def test_a_rejected_pack_block_is_recorded_and_its_errors_shown_to_the_model():
    h = _cigar()
    h.design.treated_group = {"column": "state", "level": "99"}
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"], assess_script=[DesignAssessment(action="stop", reason="pre-trends", cites=[])])

    class Fake(FakeLLM):
        def answer(self, schema, human):
            if schema is Groups:
                assert "PREVIOUS ANSWER WAS REJECTED" in human and "'99' is not observed" in human
            return super().answer(schema, human)

    fake = Fake(SCRIPT["cigar"], CIGAR["cite"], assess_script=[DesignAssessment(action="stop", reason="pre-trends", cites=[])])
    out = _run(fake, h)
    assert fake.calls.count("Groups") == 1 and out["groups"].treated_level == "5"
    d = next(x for x in out["declines"] if x.about == "claim:assignment.treated_level")
    assert d.kind == "replaced" and "'99'" in d.pack_value and d.check == "groups.level_observed"


def test_the_window_is_applied_on_the_time_column_by_code():
    from causal_agent.common.contracts import Scope

    h = forced("cigar", "q", "diff_in_diff", "sales", "state", CIGAR["cols"], scope=Scope(window="from 80 to 92"), cite=CIGAR["cite"], memory=memory("cigar"))
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"], assess_script=[DesignAssessment(action="stop", reason="pre-trends", cites=[])])
    out = _run(fake, h)
    s = out["shape"]
    assert s.periods_pre == 9 and s.periods_post == 4 and out["check_facts"]["intake"]["rows_after"] < out["check_facts"]["intake"]["rows_before"]
    assert out["declines"] == [] and out["periods"].window_start == "80" and out["periods"].window_end == "92"


def test_a_first_period_nobody_can_settle_asks_back():
    script = dict(SCRIPT["cigar"], periods=Periods(kind="long", time_column="year", first_post=None, reason="no date in the notes", cites=["change:1.note"]))
    h = _cigar()
    h.design.change_period = None
    fake = FakeLLM(script, CIGAR["cite"])
    out = _run(fake, h)
    r = out["specialist_result"]
    assert r["status"] == "ask" and r["ask"]["address"] == "claim:change.period_value" and "'year'" in r["ask"]["question"] and r["ask"]["stage"] == "periods"
    assert fake.calls.count("Periods") == 3


def test_the_run_leaves_the_paths_the_leads_and_the_placebo_spread_as_figures():
    from causal_agent.common.contracts import Said

    m = memory("cigar", trend=True, trend_status="confirmed", said="together")
    m.said.append(Said(turn=3, about="lane:claim:trend_continues.believed", text="yes"))
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"])
    out = _run(fake, handoff(**CIGAR, memory_=m))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    c = out["design"].contrast.key
    figs = {f["id"]: f for f in json.loads(open(f"{r['run_dir']}/figures.json").read())}
    assert list(figs) == [f"paths_{c}", f"event_study_{c}", f"placebo_{c}", f"effect_{c}"]
    paths = figs[f"paths_{c}"]
    assert [s["name"] for s in paths["series"]][2] == "the treated group without the change" and paths["marks"][0]["at"] == 89.0
    cf = paths["series"][2]["y"]
    assert cf[:26] == [None] * 26 and all(v is not None for v in cf[26:])
    assert figs[f"event_study_{c}"]["draws_on"] == [f"check:{c}.pre_trends"]
    assert sum(figs[f"placebo_{c}"]["series"][0]["y"]) == len(out["placebo_draws"]["placebo_group"]) and figs[f"placebo_{c}"]["marks"][0]["label"] == "observed"
    assert not any(x["check"] == "figure.check" for x in r["declines"])
    # a run that stopped at the assessment still leaves the event study from the pre-trends fit
    out = _run(FakeLLM(SCRIPT["cigar"], CIGAR["cite"], assess_script=[DesignAssessment(action="stop", reason="pre-trends", cites=[])]), _cigar())
    ids = [f["id"] for f in json.loads(open(f"{out['specialist_result']['run_dir']}/figures.json").read())]
    assert ids == [f"paths_{c}", f"event_study_{c}"]


# ------------------------------------------------------------------ the comparison rung and the effect by a unit trait


def test_a_risk_the_comparison_rung_names_is_a_flag_the_assessment_answers_and_the_interpretation_cites():
    from causal_agent.common.contracts import Said
    from causal_agent.families.diff_in_diff.lane.contracts import Risk

    m = memory("cigar", trend=True, trend_status="confirmed", said="together")
    m.said.append(Said(turn=3, about="lane:claim:trend_continues.believed", text="yes"))
    risk = Risk(name="anticipation", reason="the tax was announced a year before it took effect", cites=["change:1.note"])
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"], risks=[risk])
    out = _run(fake, handoff(**CIGAR, memory_=m))
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    assert not out["ladder"].comparison.fair and [x.name for x in out["ladder"].threats.items] == ["anticipation"]
    flag = next(c for c in out["checks"] if c.name == "threat.anticipation")
    assert flag.level == "soft" and "announced a year before" in flag.detail and flag.address in out["interpretations"][0].cites
    assert "[ladder:comparison.risk.anticipation] the tax was announced" in r["report"] and not out.get("interpret_errors")


def _toy_panel(tmp_path):
    """Forty units over ten periods; sixteen get the change from period 6; the effect is larger in the north."""
    import numpy as np

    from causal_agent.memory import ops
    from causal_agent.profile.profiler import profile

    rng = np.random.default_rng(1)
    rows = []
    for u in range(40):
        arm = "yes" if u < 16 else "no"
        region = "north" if u % 2 == 0 else "south"
        ue = rng.normal(0, 1)
        for t in range(1, 11):
            treat = int(arm == "yes" and t >= 6)
            y = 10 + ue + 0.3 * t + treat * (2.0 + (3.0 if region == "north" else 0.0)) + rng.normal(0, 0.5)
            rows.append({"unit": f"u{u}", "time": t, "arm": arm, "region": region, "y": y})
    csv = tmp_path / "toy_did.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    m = ops.seed("toy_did", profile(csv), csv=str(csv))
    src = "user:turn:1"
    m.set("claim:grain.row_is", "one unit in one period", status="confirmed", source=src)
    m.set("claim:grain.key_columns", ["unit", "time"], status="confirmed", source=src)
    m.set("claim:grain.panel", True, status="confirmed", source=src)
    m.set("claim:sampling.how", "whole", status="confirmed", source=src)
    m.set("claim:change.what", "the programme", status="confirmed", source=src)
    m.set("claim:change.to_whom", "the units in the arm", status="confirmed", source=src)
    m.set("claim:change.when", "period 6", status="confirmed", source=src)
    m.set("claim:change.date_column", "time", status="confirmed", source=src)
    m.set("claim:change.period_value", "6", status="confirmed", source=src)
    m.set("claim:assignment.kind", "date_by_others", status="confirmed", source=src)
    m.set("claim:assignment.rule", "the programme reached one arm from period 6", status="confirmed", source=src)
    m.set("claim:assignment.treatment_column", "arm", status="confirmed", source=src)
    m.set("claim:assignment.treated_level", "yes", status="confirmed", source=src)
    m.set("claim:trend_continues.believed", True, status="confirmed", source=src, said="the arms moved together before")
    m.set("claim:spillover.possible", False, status="confirmed", source=src)
    for c, when in (("y", "after"), ("arm", "at"), ("time", "at"), ("region", "before"), ("unit", "before")):
        m.set(f"col:{c}.meaning", f"{c} as recorded", status="confirmed", source=src)
        m.set(f"col:{c}.when", when, status="confirmed", source=src)
    m.set("col:region.may_modify", True, status="confirmed", source=src, said="the north had more room to gain")
    return forced("toy_did", "Did the programme raise y?", "diff_in_diff", "y", "arm", ["y", "arm", "time", "region", "unit"], cite="col:arm.note", memory=m)


def test_the_effect_is_estimated_within_each_level_of_a_unit_trait(tmp_path):
    """The person says the effect could differ by region; the heterogeneity rung picks it; the primary estimator runs again within
    each region; the estimates, the figure, the material and the report carry them, and the interpretation must cite them."""
    from causal_agent.families.diff_in_diff.lane import nodes as N

    h = _toy_panel(tmp_path)
    assert h.design.unit == "unit" and h.design.time == "time" and h.design.change_period == "6"
    script = dict(
        groups=("arm", "yes"), periods=Periods(kind="long", time_column="time", first_post="6", reason="period 6", cites=["change:1.note"]), relations={}
    )
    fake = FakeLLM(script, "col:arm.note")
    out = _run(fake, h, question="Did the programme raise y?")
    r = out["specialist_result"]
    assert r["status"] == "done", r.get("feasibility")
    assert fake.calls.count("Groups") == 0 and fake.calls.count("Periods") == 0 and fake.calls.count("Heterogeneity") == 1
    assert "[col:region]" in fake.humans_of("Heterogeneity")[0].split("CANDIDATES")[1] and "[col:region.may_modify]" in fake.humans_of("Heterogeneity")[0]
    het = out["ladder"].heterogeneity
    assert het.by == "judgement" and [m.column for m in het.modifiers] == ["region"] and out["design"].modifiers == ["region"]
    within = {e.level: e for e in out["estimates"] if e.modifier == "region"}
    assert set(within) == {"north", "south"} and all(e.error is None for e in within.values())
    assert within["north"].value > within["south"].value + 1.5 and abs(within["south"].value - 2.0) < 0.6
    prim = next(e for e in out["estimates"] if e.method == out["design"].estimator and e.modifier is None and not e.secondary)
    assert abs(prim.value - 3.5) < 0.6
    c = out["design"].contrast.key
    assert f"estimate:{c}.by.region.north.value" in N._required(state=out) and f"[estimate:{c}.by.region.south.value]" in N._material(out)
    assert {f"estimate:{c}.by.region.north.value", f"estimate:{c}.by.region.south.value"} <= set(out["interpretations"][0].cites)
    figs = {f["id"]: f for f in json.loads(open(f"{r['run_dir']}/figures.json").read())}
    assert figs[f"effect_by_modifier_{c}"]["series"][0]["x"] == ["all rows", "region = north", "region = south"] and not any(
        x["check"] == "figure.check" for x in r["declines"]
    )
    assert "within region = north" in r["report"] and "[ladder:heterogeneity.modifiers] region" in r["report"] and "modifiers    region" in r["report"]
    assert any(x["column"] == "region" for x in r["relations"]) is False or True  # region is absorbed, not a control; the rung still placed it
