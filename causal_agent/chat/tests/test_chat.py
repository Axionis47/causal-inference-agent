"""The desk with a fake model and a canned pipeline. No Vertex calls, nothing written under data/."""

from __future__ import annotations

import importlib.util
import json
import re
import uuid
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from causal_agent.chat import material as M
from causal_agent.chat import nodes as N
from causal_agent.chat import pipeline
from causal_agent.chat.contracts import AfterReply, NumberStated, RunRecord
from causal_agent.chat.graph import compile_local
from causal_agent.common.llm import set_llm
from causal_agent.profile import data as D
from causal_agent.intake.interview import writer as W
from causal_agent.memory.claims import Extraction, Reply

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
FIX = HERE / "fixtures"
STUDENTS = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"

# the interview's scripted fake and helpers, one source
_spec = importlib.util.spec_from_file_location("interview_tests", ROOT / "causal_agent/intake/interview/tests/test_interview.py")
IT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(IT)


def canned(lane: str, index: int, question: str, effect: float | None = None) -> RunRecord:
    a = json.loads((FIX / lane / "artifacts.json").read_text())
    d = json.loads((FIX / lane / "design.json").read_text())
    family, spec = {"dowhy": ("adjustment", "dowhy"), "did": ("diff_in_diff", "pyfixest"), "rd": ("discontinuity", "rdrobust")}[lane]
    if effect is not None:
        for e in a["estimates"]:
            if not e.get("secondary"):
                e["value"] = effect
    sr = {"status": "done", "family": family, "specialist": spec, "run_dir": None, "design": d, "estimates": a["estimates"], "refutations": a["refutations"],
          "interpretations": a["interpretations"], "feasibility": None}
    out = {"specialist_result": sr, "handoff": {"family": family, "specialist": spec, "chosen_assumption": "the arms are alike given what was measured"},
           "decision": {"chosen": family, "chosen_assumption": "the arms are alike given what was measured", "why_over_alternatives": "only admissible family",
                        "rejected": [{"family": "diff_in_diff", "reason": "no periods"}]}, "decision_record": "CHOSEN " + family}
    rec = pipeline.record("students2", question, index, out)
    rec.artifacts = a
    return rec


class Fake(IT.FakeLLM):
    """The interview fake plus a queue of AfterReply objects."""

    def __init__(self, *a, after=None, **k):
        super().__init__(*a, **k)
        self.after = list(after or [])
        self.after_humans: list[str] = []

    def answer(self, schema, human):
        if schema is AfterReply:
            self.calls.append("AfterReply")
            self.after_humans.append(human)
            return self.after.pop(0)
        return super().answer(schema, human)


@pytest.fixture(autouse=True)
def _restore(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    yield
    set_llm(None)
    D.clear()


def values(g, cfg) -> dict:
    """Parent values, overlaid with the interview subgraph's values while it is the one interrupted."""
    s = g.get_state(cfg, subgraphs=True)
    v = dict(s.values)
    for t in s.tasks:
        sub = getattr(t, "state", None)
        if sub is not None and hasattr(sub, "values"):
            v.update(sub.values)
    return v


def start(fake, csv, doc, question, name="students2"):
    set_llm(fake)
    g = compile_local()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    payload = None
    for mode, chunk in g.stream({"dataset": name, "csv": str(csv), "docs": {"context": doc}, "question": question}, cfg, stream_mode=["updates"]):
        if "__interrupt__" in chunk:
            payload = chunk["__interrupt__"][0].value
    return g, cfg, payload


def say(g, cfg, text):
    payload = None
    for mode, chunk in g.stream(Command(resume=text), cfg, stream_mode=["updates"]):
        if "__interrupt__" in chunk:
            payload = chunk["__interrupt__"][0].value
    return payload, values(g, cfg)


# ------------------------------------------------------------------ material


@pytest.mark.parametrize("lane", ["dowhy", "did", "rd"])
def test_material_resolves_every_cite_the_lane_made(lane):
    rec = canned(lane, 1, "q")
    mat = M.render(rec, None)
    for i in rec.artifacts["interpretations"]:
        missing = [c for c in i["cites"] if c not in mat.addresses]
        assert not missing, missing
    prim = next(e for e in rec.artifacts["estimates"] if not e.get("secondary"))
    c = prim["contrast"]
    assert abs(mat.numbers[f"estimate:{c}.value"] - prim["value"]) < 1e-9
    assert rec.effect == prim["value"] and rec.family
    text = M.brief(rec, None, mat)
    assert f"[estimate:{c}.value]" in text and "Reading:" in text


# ------------------------------------------------------------------ the gate


def test_gate_grounds_every_number_and_cite():
    rec = canned("dowhy", 1, "q")
    mat = M.render(rec, None)
    c = "completed_vs_none"
    ok = AfterReply(kind="answer", text="The course raised scores by about 5.618 points, interval 3.685 to 7.55.", cites=[f"estimate:{c}.value", f"estimate:{c}.ci"],
                    numbers=[NumberStated(address=f"estimate:{c}.value", value=5.618), NumberStated(address=f"estimate:{c}.ci_low", value=3.685), NumberStated(address=f"estimate:{c}.ci_high", value=7.55)])
    assert N._gate(ok, mat, {}) == []
    wrong = ok.model_copy(update={"numbers": [NumberStated(address=f"estimate:{c}.value", value=6.1)], "text": "The effect is 6.1 points."})
    errs = N._gate(wrong, mat, {})
    assert any("does not match" in e for e in errs)
    loose = ok.model_copy(update={"text": "The effect is 5.618 and the sample had 1234 rows.", "numbers": ok.numbers})
    assert any("1234" in e for e in N._gate(loose, mat, {}))
    # a number quoted from a check's own detail text is grounded by that line even when it is not the check's headline value
    rec_rd = canned("rd", 1, "q")
    mat_rd = M.render(rec_rd, None)
    dens = next(a for a in mat_rd.addresses if a.endswith(".density"))
    quoted = float(re.findall(r"threshold ([\d.]+)", mat_rd.by_address[dens])[0])
    fromline = AfterReply(kind="answer", text=f"The density test's threshold is {quoted:g}.", cites=[dens], numbers=[NumberStated(address=dens, value=quoted)])
    assert N._gate(fromline, mat_rd, {}) == []
    badcite = ok.model_copy(update={"cites": ["estimate:nope.value"]})
    assert any("cites not in the material" in e for e in N._gate(badcite, mat, {}))
    assert any("revise needs" in e for e in N._gate(AfterReply(kind="revise", text="ok"), mat, {}))
    assert any("requestion needs" in e for e in N._gate(AfterReply(kind="requestion", text="ok"), mat, {}))
    assert N._gate(AfterReply(kind="done", text="bye"), mat, {}) == []


# ------------------------------------------------------------------ end to end


def test_desk_interview_run_answer_revise_rerun_requestion(tmp_path, monkeypatch):
    calls = []

    def fake_run(dataset, question, index):
        calls.append((dataset, question, index))
        return canned("dowhy", index, question, effect={1: 5.618, 2: 4.2, 3: 1.1}[index])

    monkeypatch.setattr(pipeline, "run", fake_run)
    c = "completed_vs_none"
    after = [
        # a first answer that states a number the artifacts do not hold, then a grounded one
        AfterReply(kind="answer", text="It raised scores by 6.1 points.", cites=[f"estimate:{c}.value"], numbers=[NumberStated(address=f"estimate:{c}.value", value=6.1)]),
        AfterReply(kind="answer", text="It raised math scores by 5.618 points.", cites=[f"estimate:{c}.value"], numbers=[NumberStated(address=f"estimate:{c}.value", value=5.618)]),
        # a design wish answered as a claim, not a revision
        AfterReply(kind="answer", text="The estimator follows from the claims; lunch would leave the adjustment set only if it was set after the course.", cites=["claim:col:lunch"]),
        # a revision
        AfterReply(kind="revise", text="I will re-check with lunch set after the course.", claim_updates=[IT.U("measured", ["user:turn:9"], column="lunch", meaning="lunch status", when="after")]),
        # after the rerun: then and now
        AfterReply(kind="answer", text="It moved from 5.618 to 4.2.", cites=["run:1.effect", f"estimate:{c}.value"],
                   numbers=[NumberStated(address="run:1.effect", value=5.618), NumberStated(address=f"estimate:{c}.value", value=4.2)]),
        AfterReply(kind="requestion", text="Running it for reading scores.", question="Did completing the prep course raise reading scores?"),
        AfterReply(kind="done", text="Bye."),
    ]
    fake = Fake({"doc:context": IT.students_turn0()}, after=after)
    g, cfg, payload = start(fake, STUDENTS, IT.STUDENTS_DOC, "Did completing the prep course raise math scores?")
    assert payload["ready"] is False  # interview asks
    v = values(g, cfg)
    fake.script["user:turn:1"] = IT.confirm_all(v["claims"], "user:turn:1")
    payload, v = say(g, cfg, "all right")
    fake.script["user:turn:2"] = [IT.U("unobserved", ["user:turn:2"], exists="false"), IT.U("spillover", ["user:turn:2"], possible="false"), IT.U("exclusion", ["user:turn:2"], exists="false")]
    payload, v = say(g, cfg, "nothing hidden, no spillover, no nudge")
    assert payload["ready"] and v["status"].ready
    # run
    payload, v = say(g, cfg, "run")
    assert calls == [("students2", "Did completing the prep course raise math scores?", 1)]
    assert payload["phase"] == "after" and "5.618" in payload["text"] and f"[estimate:{c}.value]" in payload["text"]
    assert v["runs"][0].effect == 5.618 and v["written"]
    # answer: the bad number is rejected once, then the good one passes
    payload, v = say(g, cfg, "what did you find?")
    assert fake.calls.count("AfterReply") == 2 and v["after_errors"] == [] and "5.618" in payload["text"]
    assert "does not match" in fake.after_humans[1]
    # a design wish is answered, not revised
    payload, v = say(g, cfg, "use the quadratic instead")
    assert v["after_reply"].kind == "answer" and len(v["runs"]) == 1
    # revise: the interview re-enters with the claims kept, the person confirms, run again
    fake.dynamic = lambda src, human: [] if src.startswith("user:turn:") else None
    payload, v = say(g, cfg, "actually lunch was set after the course")
    assert v["phase"] == "before" and v["claims"].get("col:lunch").fields["when"] == "after" and v["claims"].get("col:lunch").status == "confirmed"
    assert v["claims"].get("assignment").status == "confirmed"  # kept, not reseeded
    assert payload["ready"], payload
    payload, v = say(g, cfg, "run")
    assert len(v["runs"]) == 2 and calls[-1][2] == 2 and "Then and now" in payload["text"] and "4.2" in payload["text"]
    payload, v = say(g, cfg, "did the effect change?")
    assert v["after_errors"] == [] and "4.2" in payload["text"]
    # requestion
    payload, v = say(g, cfg, "same thing for reading scores")
    assert len(v["runs"]) == 3 and calls[-1][1].startswith("Did completing the prep course raise reading") and v["runs"][-1].effect == 1.1
    payload, v = say(g, cfg, "thanks, done")
    assert payload is None and len(v["exchanges"]) >= 5
    assert all(t.thinking_tokens == 7 for t in v["debug"])


def test_quit_before_run_ends_without_running(monkeypatch):
    monkeypatch.setattr(pipeline, "run", lambda *a: (_ for _ in ()).throw(AssertionError("ran")))
    fake = Fake({"doc:context": IT.students_turn0()})
    g, cfg, payload = start(fake, STUDENTS, IT.STUDENTS_DOC, "q")
    payload, v = say(g, cfg, "quit")
    assert payload is None and not v.get("runs")


def test_pipeline_record_from_the_subprocess_json(tmp_path):
    """The router subprocess writes handoff, decision, record, and result as JSON; record() reads that shape."""
    a = json.loads((FIX / "rd" / "artifacts.json").read_text())
    d = json.loads((FIX / "rd" / "design.json").read_text())
    out = {"handoff": {"family": "discontinuity", "specialist": "rdrobust", "chosen_assumption": "alike at the cutoff"},
           "decision": {"chosen": "discontinuity", "chosen_assumption": "alike at the cutoff", "why_over_alternatives": "the cutoff identifies", "rejected": [{"family": "adjustment", "reason": "strict rule"}]},
           "decision_record": "CHOSEN discontinuity", "gate_errors": [],
           "specialist_result": {"status": "done", "family": "discontinuity", "specialist": "rdrobust", "run_dir": None, "design": d, "estimates": a["estimates"], "refutations": a["refutations"], "interpretations": a["interpretations"], "feasibility": None}}
    rec = pipeline.record("senate2", "q", 1, out)
    assert rec.family == "discontinuity" and rec.status == "done" and abs(rec.effect - 7.3948) < 1e-3 and rec.decision["over"] == {"adjustment": "strict rule"}
    mat = M.render(rec, None)
    assert "decision.over:adjustment" in mat.addresses and "design.bandwidth" in mat.addresses
    err = RunRecord(index=1, dataset="x", question="q", status="pipeline_error", decision_record="boom")
    assert "failed before producing a record" in M.brief(err, None, M.render(err, None))
