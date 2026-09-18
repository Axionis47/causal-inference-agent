"""The diff-in-diff lane with a fake model and real pyfixest on the real files. No Vertex calls."""

from __future__ import annotations

import re
import uuid

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from causal_agent.common.contracts import Cited, Handoff, Interpretation
from causal_agent.common.llm import set_llm
from causal_agent.desk.handoff import forced
from causal_agent.specialists.did.contracts import ControlRelation, DesignAssessment, EstimatorPick, Groups, Periods, Revision
from causal_agent.specialists.did.graph import compile_local


def handoff(pack: str, outcome: str, treatment: str, cols: list[str], cite: str) -> Handoff:
    return forced(pack, "q", "diff_in_diff", outcome, treatment, cols, assumption="parallel movement absent the change", cite=cite)


CK = dict(pack="card_krueger", outcome="total_emp_nov", treatment="state", cols=["state", "total_emp_feb", "total_emp_nov"], cite="col:state.note")
CIGAR = dict(pack="cigar", outcome="sales", treatment="state", cols=["state", "year", "sales", "price", "pimin", "ndi", "pop", "cpi"], cite="col:state.note")
MARKETING = dict(pack="marketing", outcome="SalesInThousands", treatment="Promotion", cols=["Promotion", "week", "SalesInThousands", "MarketSize", "LocationID"], cite="col:promotion.note")

# scripted answers per dataset, what a careful reader of the notes would say
SCRIPT = {
    "card_krueger": dict(groups=("state", "1"), periods=Periods(kind="wide", before_column="total_emp_feb", after_column="total_emp_nov", reason="the notes name the two waves", cites=["col:total_emp_feb.note"]),
                         relations={}),
    "cigar": dict(groups=("state", "5"), periods=Periods(kind="long", time_column="year", first_post="89", reason="Proposition 99 from January 1989", cites=["change:1.note"]),
                  relations={"price": dict(affected_by_treatment=True), "pimin": dict(usable_as_control=True), "ndi": dict(usable_as_control=True), "pop": dict(usable_as_control=True), "cpi": dict(usable_as_control=True)}),
    "marketing": dict(groups=("promotion", "2"), periods=Periods(kind="long", time_column="week", first_post="1", reason="every row is under the promotion", cites=["change:1.note"]),
                      relations={"marketsize": dict(usable_as_control=True)}),
}


class FakeLLM:
    def __init__(self, script: dict, cite: str, *, bad_cites=False, assess_script=None, pick_script=None):
        self.script, self.cite, self.bad_cites = script, cite, bad_cites
        self.assess_script, self.pick_script = list(assess_script or []), list(pick_script or [])
        self.calls: list[str] = []

    def with_structured_output(self, schema, include_raw=False):
        fake = self

        class R:
            def invoke(self_, messages):
                parsed = fake.answer(schema, messages[-1][1])
                raw = AIMessage(content=[{"type": "thinking", "thinking": f"thinking about {schema.__name__}"}, "{}"],
                                usage_metadata={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30, "output_token_details": {"reasoning": 7}})
                return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return R()

    def answer(self, schema, human):
        self.calls.append(schema.__name__)
        cite = "col:nope.note" if self.bad_cites else self.cite
        if schema is Groups:
            col, lvl = self.script["groups"]
            return Groups(column=col, treated_level=lvl, reason="the change card says so", cites=[cite])
        if schema is Periods:
            p = self.script["periods"].model_copy()
            p.cites = [cite]
            return p
        if schema is ControlRelation:
            col = re.search(r"for column '([^']+)'", human).group(1)
            flags = dict(affected_by_treatment=False, usable_as_control=False)
            flags.update(self.script["relations"].get(col, {}))
            return ControlRelation(column=col, reasons=[Cited(reason=f"{col}: {k}", cites=[cite]) for k, v in flags.items() if v], **flags)
        if schema is DesignAssessment:
            if self.assess_script:
                return self.assess_script.pop(0)
            return DesignAssessment(action="proceed", reason="only soft flags", cites=[])
        if schema is EstimatorPick:
            names = [n.strip() for n in re.search(r"NAMES YOU MAY PICK: (.*)", human).group(1).split(",")]
            return EstimatorPick(name=self.pick_script.pop(0) if self.pick_script else names[0], reason="ranked first", cites=[])
        if schema is Interpretation:
            addresses = human.split("ADDRESSES YOU MAY CITE")[1].split("\n\n")[0].strip().splitlines()
            contrast = re.search(r"COMPARISON: (\S+)", human).group(1)
            value = float(re.search(r"\[estimate:%s\.value\] ([-\d.eE+]+)" % re.escape(contrast), human).group(1))
            return Interpretation(contrast=contrast, answer=f"The effect on the treated is {value:.3g}.", effect_stated=value, caveats=["parallel trends assumed"], cites=addresses[:3])
        raise AssertionError(schema)


@pytest.fixture(autouse=True)
def _restore(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_DIR", str(tmp_path / "runs"))
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
    assert {c.name for c in out["checks"] if c.level == "soft"} == {"parallel_untestable"}
    assert [x.refuter for x in out["refutations"]] == ["placebo_group"] and out["refutations"][0].passed is True
    assert fake.calls.count("ControlRelation") == 0 and "DesignAssessment" in fake.calls
    assert len(out["interpretations"]) == 1 and not out.get("interpret_errors")
    assert "DESIGN" in r["report"] and "ANSWER" in r["report"]


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
    assert fake.calls.count("ControlRelation") == 5


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
    assert fake.calls.count("ControlRelation") == 0


def test_bad_cites_loop_relate_then_stop():
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"], bad_cites=True)
    out = _run(fake, handoff(**CIGAR))
    assert out["specialist_result"]["status"] == "infeasible"
    # groups and periods have their own in-node retries (3 each) before the relate loop can even start
    assert out["feasibility"].stage in ("groups", "periods", "verify")


def test_relate_bad_cites_only():
    class Fake(FakeLLM):
        def answer(self, schema, human):
            if schema is ControlRelation:
                self.bad_cites = True
            else:
                self.bad_cites = False
            return super().answer(schema, human)

    fake = Fake(SCRIPT["cigar"], CIGAR["cite"])
    out = _run(fake, handoff(**CIGAR))
    assert out["feasibility"].stage == "verify"
    assert fake.calls.count("ControlRelation") == 5 * 3


def test_revision_is_a_delta():
    script = [DesignAssessment(action="revise", reason="drop pop", cites=[], revisions=[Revision(column="pop", change="remove_control", reason="test", cites=["col:pop.note"])]),
              DesignAssessment(action="stop", reason="pre-trends still fail", cites=[])]
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"], assess_script=script)
    out = _run(fake, handoff(**CIGAR))
    assert out["revisions"] == 1 and "pop" not in out["controls"].included
    assert fake.calls.count("ControlRelation") == 5  # no worker reran


def test_estimator_outside_list_is_rejected_then_accepted():
    fake = FakeLLM(SCRIPT["card_krueger"], CK["cite"], pick_script=["magic", "twfe_static"])
    out = _run(fake, handoff(**CK))
    assert out["specialist_result"]["status"] == "done" and fake.calls.count("EstimatorPick") == 2


def test_catalogues_parse_on_a_toy_panel():
    import pyfixest as pf

    from causal_agent.specialists.did import adapter
    from causal_agent.specialists.did.knowledge import load_checks, load_estimators, load_inference, load_placebos

    rng = pd.Series(range(200))
    toy = pd.DataFrame({"unit": (rng % 20).astype(str), "time": rng // 20})
    toy["treated"] = (toy["unit"].astype(int) < 8).astype(int)
    toy["post"] = (toy["time"] >= 5).astype(int)
    toy["treat"] = (toy["treated"] * toy["post"]).astype(float)
    toy["rel_time"] = toy["time"] - 5
    toy["cohort"] = toy["treated"] * 5
    toy["x1"] = (toy["time"] * 0.3 + toy["unit"].astype(int) * 0.1)
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


def test_router_wires_both_specialists():
    from causal_agent.router.graph import graph as router_graph
    from causal_agent.specialists import SPECIALISTS

    assert "shape_table" in SPECIALISTS["diff_in_diff"].get_graph().nodes
    assert "freeze_design" in SPECIALISTS["adjustment"].get_graph().nodes and "shape_table" not in SPECIALISTS["adjustment"].get_graph().nodes
    assert len(router_graph.get_graph().nodes) == 16


# ------------------------------------------------------------------ the pack's facts end judgements


def test_pack_panel_block_settles_groups_and_periods_without_a_model_call():
    """No shipped dataset carries diff-in-diff claims yet, so the claims are built here and handed to the builder directly."""
    from causal_agent.common.contracts import Candidate, FamilyDecision, QuestionFrame, Scope
    from causal_agent.desk.handoff import build
    from causal_agent.profile.datasets import load_dataset_pack
    from causal_agent.memory.claims import Claim, ClaimTable
    from causal_agent.knowledge import load_registry

    table = ClaimTable(claims={
        "grain": Claim(kind="grain", key="grain", fields={"row_is": "one state in one year", "key_columns": ["state", "year"], "panel": True}, status="confirmed", source="user:turn:1"),
        "change": Claim(kind="change", key="change", fields={"what": "Proposition 99", "to_whom": "California", "when": "January 1989", "date_column": "year", "period_value": "89"}, status="confirmed", source="user:turn:1"),
        "assignment": Claim(kind="assignment", key="assignment", fields={"kind": "date_by_others", "rule": "a ballot vote in one state", "treatment_column": "state", "treated_level": "5"}, status="confirmed", source="user:turn:1"),
        "trend_continues": Claim(kind="trend_continues", key="trend_continues", fields={"believed": True, "why": "sales moved together before 1989"}, status="confirmed", source="user:turn:2"),
    })
    cols = ["sales", "state", "year", "price", "pimin", "ndi", "pop", "cpi"]
    frame = QuestionFrame(intent="effect_of_change", decision_served="", outcome_candidates=[Candidate(column="sales", reason="r", cites=["col:sales.note"])],
                          cause_candidates=[Candidate(column="state", reason="r", cites=["col:state.note"])], scope=Scope(target="on_treated"),
                          relevant_columns=[Candidate(column=c, reason="r", cites=["col:state.note"]) for c in cols], reasons=[])
    decision = FamilyDecision(admissible=["diff_in_diff"], chosen="diff_in_diff", chosen_assumption="parallel movement absent the change", why_over_alternatives="only one", rejected=[])
    fam = next(f for f in load_registry() if f.name == "diff_in_diff")
    h = build(question="q", frame=frame, decision=decision, family=fam, pack=load_dataset_pack("cigar"), claims=table)
    d = h.design
    assert d.kind == "diff_in_diff" and d.unit == "state" and d.time == "year" and d.change_period == "89" and d.treated_group == {"column": "state", "level": "5"}
    assert d.trend_belief is not None and d.trend_belief.value is True
    assert h.resolve("claim:change.period_value") and h.resolve("claim:trend_continues")
    fake = FakeLLM(SCRIPT["cigar"], CIGAR["cite"], assess_script=[DesignAssessment(action="stop", reason="pre-trends", cites=[])])
    out = _run(fake, h)
    assert fake.calls.count("Groups") == 0 and fake.calls.count("Periods") == 0
    assert out["groups"].column == "state" and out["groups"].treated_level == "5" and out["periods"].first_post == "89"
    assert out["groups"].cites == ["claim:assignment.treatment_column", "claim:assignment.treated_level"]
