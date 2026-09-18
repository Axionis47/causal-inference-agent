"""The interview with a fake model on real and synthetic files. No Vertex calls."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from langchain_core.messages import AIMessage
from langgraph.types import Command

from causal_agent.common.llm import set_llm
from causal_agent.intake.interview import data as D
from causal_agent.intake.interview import nodes as N
from causal_agent.intake.interview import table as T
from causal_agent.intake.interview import writer as W
from causal_agent.intake.interview.contracts import Claim, ClaimTable, ClaimUpdate, Extraction, FieldValue, ProbeResult, Question, Reply
from causal_agent.intake.interview.graph import compile_local
from causal_agent.intake.knowledge import load_catalogue, load_thresholds
from causal_agent.intake.pack import load_pack
from causal_agent.intake.profiler import profile

ROOT = Path(__file__).resolve().parents[4]
STUDENTS = ROOT / "data/raw/students-performance-in-exams/StudentsPerformance.csv"
CAT = load_catalogue()

STUDENTS_DOC = """Each row is one student's results from the May 2026 exam at one school; every student who sat is included.
The school ran a six-week prep course before the exam; places were offered first to free-lunch students and to those whose
parents hold no degree, then to anyone who asked. Completion was recorded by the counsellor.
gender: recorded at enrolment. race/ethnicity: the school's grouping, recorded at enrolment.
parental level of education: declared at enrolment. lunch: standard or free/reduced, set by the district before the exam.
test preparation course: completed or none, decided by offer and uptake before the exam.
math score, reading score, writing score: the exam marks, 0 to 100, recorded at the sitting."""


def U(claim_kind, cites, column=None, unknown=False, **values):
    return ClaimUpdate(kind=claim_kind, column=column, unknown=unknown, reason="scripted", cites=list(cites),
                       values=[FieldValue(name=k, value=str(v)) for k, v in values.items()])


def students_turn0(cite="doc:context"):
    ups = [
        U("grain", [cite], row_is="one student's exam results", panel="false"),
        U("sampling", [cite], how="whole", detail="every student who sat the exam"),
        U("change", [cite], what="a six-week test preparation course", to_whom="students at the school", when="the six weeks before the May 2026 exam"),
        U("assignment", [cite], kind="own_choice", rule="offered first by lunch status and parental education, then open to anyone who asked", treatment_column="test preparation course", treated_level="completed"),
    ]
    cols = {"gender": "before", "race/ethnicity": "before", "parental level of education": "before", "lunch": "before",
            "test preparation course": "at", "math score": "after", "reading score": "after", "writing score": "after"}
    for c, when in cols.items():
        ups.append(U("measured", [cite, "change:1"] if False else [cite], column=c, meaning=f"{c} as recorded", when=when))
    return ups


def confirm_all(table: ClaimTable, cite: str):
    """The person says every draft is right: one update per drafted claim, on their word."""
    ups = []
    for c in table.claims.values():
        if c.status in {"drafted", "refuted"}:
            ups.append(U(c.kind, [cite], column=c.key[4:] if c.kind == "measured" else None, **{k: v for k, v in c.fields.items() if v is not None and not isinstance(v, list)}))
    return ups


class FakeLLM:
    """Extraction from a script keyed by the material's source; replies built legally from the open list, unless a bad
    reply is queued first."""

    def __init__(self, script: dict[str, list[ClaimUpdate]] | None = None, *, bad_replies: list[Reply] | None = None, dynamic=None):
        self.script, self.bad_replies, self.dynamic = dict(script or {}), list(bad_replies or []), dynamic
        self.calls: list[str] = []
        self.humans: list[str] = []

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
        self.humans.append(human)
        if schema is Extraction:
            src = re.search(r"\n\[((?:doc:|user:turn:)[^\]]+)\] ", human).group(1)
            if "THE MATERIAL, AGAIN" in human:  # the focused second pass: nothing new in a scripted fake
                return Extraction(updates=[])
            if self.dynamic:
                ups = self.dynamic(src, human)
                if ups is not None:
                    return Extraction(updates=ups)
            return Extraction(updates=self.script.get(src, []))
        if schema is Reply:
            if self.bad_replies:
                return self.bad_replies.pop(0)
            return legal_reply(human)
        raise AssertionError(schema)


def parse_open(human: str) -> list[tuple[str, str, str]]:
    sec = human.split("STILL OPEN, IN THE ORDER TO ASK")[1].split("THE PERSON'S LAST MESSAGE")[0]
    return re.findall(r"^\[claim:([^\]]+)\] (\w+) (\w+)", sec, flags=re.M)


def legal_reply(human: str) -> Reply:
    qs = []
    measured = [(k, st) for k, kind, st in parse_open(human) if kind == "measured"]
    for key, kind, st in parse_open(human):
        if kind == "measured":
            continue
        spec = CAT.kinds[kind]
        ev = re.findall(r"\[(check:%s\.[^\]]+)\]" % re.escape(key), human)
        if st in {"drafted", "refuted"}:
            qs.append(Question(keys=[key], kind="confirm", text=f"I read {kind} from your note. Is that right?", evidence_cites=ev[-1:]))
        else:
            field = next(n for n, s in spec.fields.items() if not s.optional)
            opts = N._legal_options(spec, field)
            if opts:
                qs.append(Question(keys=[key], field=field, kind="choose", text=f"{spec.frame}? ({', '.join(opts)})", options=opts))
            else:
                qs.append(Question(keys=[key], field=field, kind="open", text=f"{spec.frame}?"))
    if measured:
        drafted = [k for k, st in measured if st in {"drafted", "refuted"}]
        empty = [k for k, st in measured if st not in {"drafted", "refuted"}]
        if drafted:
            qs.append(Question(keys=drafted, field="when", kind="confirm", text="I read these columns' timing from your lines. Any wrong?", evidence_cites=[f"claim:{k}" for k in drafted]))
        if empty:
            qs.append(Question(keys=empty, field="when", kind="choose", text="For the rest: fixed before the change, set at it, or measured after?", options=["before", "at", "after", "unknown"]))
    return Reply(questions=qs, text="\n".join(q.text for q in qs) or "All settled.")


@pytest.fixture(autouse=True)
def _restore(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    yield
    set_llm(None)
    D.clear()


def start(fake, csv, doc, name="ds"):
    set_llm(fake)
    g = compile_local()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    g.invoke({"dataset": name, "csv": str(csv), "docs": {"context": doc}, "question": None}, cfg)
    return g, cfg


def say(g, cfg, text):
    g.invoke(Command(resume=text), cfg)
    return g.get_state(cfg).values


def synthetic_cutoff(tmp_path, n=400, side="below", takeup=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    treated = (x < 0) if side == "below" else (x > 0)
    t = np.where(treated, rng.binomial(1, takeup, n), 0)
    y = 1.0 + 2.0 * t + 0.5 * x + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"hh": range(n), "score": x, "got_it": t, "outcome": y, "age": rng.integers(20, 70, n)})
    p = tmp_path / "cut.csv"
    df.to_csv(p, index=False)
    return p


def cutoff_turn0(cite="doc:context", side="below"):
    return [
        U("grain", [cite], row_is="one household", key_columns="hh", panel="false"),
        U("sampling", [cite], how="whole"),
        U("change", [cite], what="a cash transfer", to_whom="poor households", when="2006"),
        U("assignment", [cite], kind="cutoff_rule", rule="households with a score below zero were eligible", score_column="score", cutoff="0", treated_side=side,
          cutoff_value_treated="false", treatment_column="got_it", treated_level="1", movable="false"),
        U("measured", [cite], column="hh", meaning="household id", when="before"),
        U("measured", [cite], column="score", meaning="the eligibility score", when="before"),
        U("measured", [cite], column="got_it", meaning="received the transfer", when="at"),
        U("measured", [cite], column="outcome", meaning="support for the government", when="after"),
        U("measured", [cite], column="age", meaning="age of the head", when="before"),
    ]


# ------------------------------------------------------------------ the happy path


def test_students_drafts_then_settles_then_writes(tmp_path):
    fake = FakeLLM({"doc:context": students_turn0()}, dynamic=lambda src, human: None)
    g, cfg = start(fake, STUDENTS, STUDENTS_DOC, name="students2")
    v = g.get_state(cfg).values
    st = v["status"]
    assert v["claims"].get("assignment").status == "drafted"
    assert v["claims"].get("col:lunch").fields["when"] == "before"
    assert not st.ready and "assignment" in st.open and "col:lunch" in st.open
    # uncheckable claims are not asked until the assignment is settled
    assert "unobserved" not in st.open and "spillover" not in st.open
    assert st.table["discontinuity"]["assignment"] == "does_not_fit" and "discontinuity" in st.struck
    assert st.table["diff_in_diff"]["grain"] == "does_not_fit"
    assert "adjustment" in st.surviving
    reply = v["reply"]
    assert reply.questions and reply.questions[0].keys == ["assignment"] and reply.questions[0].kind == "confirm"
    assert not v["respond_errors"]

    # turn 1: the person confirms every draft; the extract prompt shows the questions they are answering
    fake.script["user:turn:1"] = confirm_all(v["claims"], "user:turn:1")
    v = say(g, cfg, "all of that is right")
    last_extract = [h for h in fake.humans if "NEW MATERIAL" in h][-1]
    assert "THE QUESTIONS THE PERSON IS ANSWERING" in last_extract and "- about assignment" in last_extract
    st = v["status"]
    assert v["claims"].get("assignment").status == "confirmed"
    assert set(st.open) == {"unobserved", "spillover", "exclusion"}
    assert not st.ready
    # turn 2: the uncheckable claims
    fake.script["user:turn:2"] = [U("unobserved", ["user:turn:2"], exists="false", why_believed="places were offered on lunch and parental education, both in the file"),
                                  U("spillover", ["user:turn:2"], possible="false"), U("exclusion", ["user:turn:2"], exists="false")]
    v = say(g, cfg, "nothing outside the file decided it; students sit alone; no separate nudge")
    st = v["status"]
    assert st.ready and st.open == [] and st.table["instrument"]["exclusion"] == "does_not_fit"
    assert v["reply"].questions == [] and "run" in v["reply"].text
    # run: write the pack
    v = say(g, cfg, "run")
    w = v["written"]
    assert v["handoff_ready"] and w["dataset"] == "students2"
    note = (tmp_path / w["note"]).read_text()
    assert "## About the dataset" in note and "## What changed" in note and "**lunch**" in note
    pack = load_pack("students2", tmp_path / w["note"], tmp_path / w["profile"], tmp_path / w["claims"])
    assert pack.unnoted_columns == [] and pack.resolve("claim:assignment.kind") and pack.resolve("claim:col:lunch.when")
    assert "CLAIMS SETTLED AT INTAKE" in pack.digest()
    entries = yaml.safe_load((tmp_path / "data/datasets.yaml").read_text())
    assert entries["students2"]["claims"] == w["claims"]
    assert all(t.thinking_tokens == 7 for t in v["debug"])


# ------------------------------------------------------------------ refutation and revision


def test_wrong_side_is_refuted_with_numbers_then_corrected(tmp_path):
    csv = synthetic_cutoff(tmp_path, side="below")
    fake = FakeLLM({"doc:context": cutoff_turn0(side="above")})
    g, cfg = start(fake, csv, "households above zero got the transfer")
    v = g.get_state(cfg).values
    a = v["claims"].get("assignment")
    assert a.status == "refuted" and "takeup_by_side" in a.evidence[-1] and "%" in a.check_detail
    q = next(q for q in v["reply"].questions if "assignment" in q.keys)
    assert q.kind == "confirm" and q.evidence_cites == [a.evidence[-1]]
    fake.script["user:turn:1"] = [U("assignment", ["user:turn:1"], treated_side="below")]
    v = say(g, cfg, "sorry, below zero")
    a = v["claims"].get("assignment")
    assert a.status == "confirmed" and a.fields["treated_side"] == "below" and "%" in a.check_detail
    assert v["status"].table["discontinuity"]["assignment"] == "fits"
    assert any(p.family == "discontinuity" and p.passed for p in v["probes"])


def test_confirmed_claim_changes_only_on_the_persons_word(tmp_path):
    csv = synthetic_cutoff(tmp_path)
    attempts = []

    def dynamic(src, human):
        if src == "user:turn:1":
            attempts.append(human)
            if "PREVIOUS UPDATES WERE REJECTED" not in human:
                return [U("assignment", ["doc:context"], treated_side="above")]  # a doc cite cannot change a confirmed claim
            return [U("assignment", ["user:turn:1"], movable="false")]
        return None

    fake = FakeLLM({"doc:context": cutoff_turn0()}, dynamic=dynamic)
    g, cfg = start(fake, csv, "x")
    v = g.get_state(cfg).values
    fake.script["user:turn:0"] = []
    # confirm on the person's word first
    fake.script["user:turn:1"] = None
    fake.dynamic = lambda src, human: confirm_all(v["claims"], "user:turn:1") if src == "user:turn:1" else None
    v = say(g, cfg, "right")
    assert v["claims"].get("assignment").status == "confirmed"
    fake.dynamic = lambda src, human: ([U("assignment", ["doc:context"], treated_side="above")] if "REJECTED" not in human else [U("assignment", ["user:turn:2"], movable="false")]) if src == "user:turn:2" else None
    v = say(g, cfg, "nobody could move it")
    assert v["claims"].get("assignment").fields["treated_side"] == "below"
    assert v["claims"].get("assignment").fields["movable"] is False
    assert fake.calls.count("Extraction") >= 3


def test_unknown_settles_and_contradiction_after_two_refutations(tmp_path):
    csv = synthetic_cutoff(tmp_path, side="below")
    fake = FakeLLM({"doc:context": cutoff_turn0(side="above")})
    g, cfg = start(fake, csv, "x")
    fake.script["user:turn:1"] = [U("assignment", ["user:turn:1"], treated_side="above")]  # insists
    v = say(g, cfg, "no, above")
    a = v["claims"].get("assignment")
    assert a.status == "contradiction" and a.refutations == 2
    assert "assignment" in v["status"].contradictions and "assignment" not in v["status"].open
    # the rest of the drafts confirmed, then don't-know on the uncheckable ones
    fake.script["user:turn:2"] = confirm_all(v["claims"], "user:turn:2")
    v = say(g, cfg, "the rest is right")
    open_now = set(v["status"].open)
    fake.script["user:turn:3"] = [U(k, ["user:turn:3"], unknown=True) for k in open_now]
    v = say(g, cfg, "I don't know")
    assert all(v["claims"].get(k).status == "unknown" for k in open_now)
    assert v["status"].ready


def test_confirmed_list_settles_drafts_and_requires_the_fields(tmp_path):
    csv = synthetic_cutoff(tmp_path)
    ups = cutoff_turn0()
    ups[0] = U("grain", ["doc:context"], row_is="one household", key_columns="hh")  # panel left unset
    fake = FakeLLM({"doc:context": ups})
    g, cfg = start(fake, csv, "x")
    v = g.get_state(cfg).values
    seen = []

    def dynamic(src, human):
        if src != "user:turn:1":
            return None
        seen.append(human)
        if "REJECTED" in human:
            return Extraction(updates=[U("grain", ["user:turn:1"], panel="false")], confirmed=[c.key for c in v["claims"].claims.values() if c.status == "drafted"])
        return Extraction(updates=[], confirmed=["claim:assignment", "change.what", "col:score.when", "grain"] + [c.key for c in v["claims"].claims.values() if c.status == "drafted"])

    fake.answer = lambda schema, human: (fake.calls.append(schema.__name__) or (Extraction(updates=[]) if "THE MATERIAL, AGAIN" in human else dynamic(re.search(r"\n\[((?:doc:|user:turn:)[^\]]+)\] ", human).group(1), human))) if schema is Extraction else legal_reply(human)
    v = say(g, cfg, "all right")
    assert len(seen) == 2 and "cannot be confirmed while ['panel'] is unset" in seen[1]
    assert v["claims"].get("assignment").status == "confirmed" and v["claims"].get("col:score").status == "confirmed"
    assert v["claims"].get("grain").status == "confirmed" and v["claims"].get("grain").fields["panel"] is False
    assert N._norm_key("claim:col:Score.when") == "col:score" and N._norm_key("trend_continues:believed") == "trend_continues"


# ------------------------------------------------------------------ the reply gate


def test_reply_gate_rejects_metric_and_method_words_then_retries(tmp_path):
    csv = synthetic_cutoff(tmp_path)
    bad = [Reply(questions=[Question(keys=["assignment"], kind="confirm", text="How many rows are treated?")], text="How many rows are treated?"),
           Reply(questions=[Question(keys=["assignment"], kind="confirm", text="Is this a regression discontinuity?")], text="Is this a regression discontinuity?")]
    fake = FakeLLM({"doc:context": cutoff_turn0()}, bad_replies=bad)
    g, cfg = start(fake, csv, "x")
    v = g.get_state(cfg).values
    assert fake.calls.count("Reply") == 3 and v["respond_errors"] == []
    assert "how many" not in v["reply"].text.lower()


def test_reply_gate_shapes(tmp_path):
    csv = synthetic_cutoff(tmp_path)
    fake = FakeLLM({"doc:context": cutoff_turn0(side="above")})
    g, cfg = start(fake, csv, "x")
    v = g.get_state(cfg).values
    table, st = v["claims"], v["status"]
    ok = legal_reply(fake.humans[-1])
    assert N._reply_gate(ok, table, st) == []
    # a question for a settled key
    r = ok.model_copy(deep=True)
    r.questions.append(Question(keys=["missing"], kind="open", text="why?"))
    assert any("settled" in e for e in N._reply_gate(r, table, st))
    # missing an open key
    r = ok.model_copy(deep=True)
    r.questions = [q for q in r.questions if "assignment" not in q.keys]
    assert any("no question for open claims" in e for e in N._reply_gate(r, table, st))
    # refuted claim asked without the check cite
    r = ok.model_copy(deep=True)
    for q in r.questions:
        if "assignment" in q.keys:
            q.evidence_cites = []
    assert any("refuted by the file" in e for e in N._reply_gate(r, table, st))
    # choose with the wrong options
    r = ok.model_copy(deep=True)
    for q in r.questions:
        if "sampling" in q.keys:
            q.kind, q.field, q.options = "choose", "how", ["whole", "some"]
    errs = N._reply_gate(r, table, st)
    assert any("must list exactly" in e for e in errs) or all(q.kind != "choose" for q in r.questions if "sampling" in q.keys)
    # confirm where there is no draft
    empty = next((k for k in st.open if table.claims[k].status == "empty"), None)
    if empty:
        r = ok.model_copy(deep=True)
        for q in r.questions:
            if empty in q.keys:
                q.kind = "confirm"
        assert any("no draft to confirm" in e for e in N._reply_gate(r, table, st))


def test_many_open_columns_are_grouped(tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({f"v{i}": rng.normal(size=60) for i in range(12)})
    df["t"] = rng.integers(0, 2, 60)
    p = tmp_path / "wide.csv"
    df.to_csv(p, index=False)
    fake = FakeLLM({"doc:context": [U("assignment", ["doc:context"], kind="lottery", rule="a draw", treatment_column="t", treated_level="1")]})
    g, cfg = start(fake, p, "x")
    v = g.get_state(cfg).values
    measured_open = [k for k in v["status"].open if k.startswith("col:")]
    assert len(measured_open) > 8
    qs = [q for q in v["reply"].questions if any(k.startswith("col:") for k in q.keys)]
    assert 1 <= len(qs) <= 3
    ten = Reply(questions=[Question(keys=[k], field="when", kind="choose", text="when?", options=["before", "at", "after", "unknown"]) for k in measured_open]
                + [q for q in v["reply"].questions if not any(k.startswith("col:") for k in q.keys)], text="when?")
    assert any("group them" in e for e in N._reply_gate(ten, v["claims"], v["status"]))


def test_first_reply_asks_assignment_and_change_first(tmp_path):
    csv = synthetic_cutoff(tmp_path)
    fake = FakeLLM({"doc:context": []})
    g, cfg = start(fake, csv, "")
    v = g.get_state(cfg).values
    assert v["status"].open[:2] == ["assignment", "change"]
    assert v["reply"].questions[0].keys == ["assignment"] and v["reply"].questions[0].kind == "choose"
    assert not any(k in v["status"].open for k in ("unobserved", "spillover", "exclusion", "trend_continues"))


def test_run_while_drafts_are_open_takes_them_on_the_persons_word(tmp_path):
    csv = synthetic_cutoff(tmp_path)
    fake = FakeLLM({"doc:context": cutoff_turn0()})
    g, cfg = start(fake, csv, "x", name="cut")
    v = g.get_state(cfg).values
    assert v["claims"].get("assignment").status == "drafted" and not v["status"].ready
    v = say(g, cfg, "run")  # drafts confirmed; the uncheckable claims are still open, so no hand-off yet
    assert v["claims"].get("assignment").status == "confirmed" and v["claims"].get("assignment").source == "user:turn:1"
    assert not v["status"].ready and set(v["status"].open) <= {"unobserved", "exclusion", "spillover", "trend_continues", "cutoff_only"}
    assert v.get("written") is None and "still open" in fake.humans[-1]
    fake.script["user:turn:2"] = [U(k, ["user:turn:2"], unknown=True) for k in v["status"].open]
    v = say(g, cfg, "I don't know any of those")
    assert v["status"].ready
    v = say(g, cfg, "run")
    assert v["written"] and v["handoff_ready"]


def test_run_on_drafts_that_settle_everything_hands_off_at_once(tmp_path):
    csv = synthetic_cutoff(tmp_path)
    ups = cutoff_turn0()
    fake = FakeLLM({"doc:context": ups})
    g, cfg = start(fake, csv, "x", name="cut2")
    v = g.get_state(cfg).values
    fake.script["user:turn:1"] = [U(k, ["user:turn:1"], unknown=True) for k in ("unobserved", "exclusion", "spillover", "trend_continues", "cutoff_only")]
    v = say(g, cfg, "no idea about the rest")  # settles the uncheckable ones; drafts remain drafted
    assert not v["status"].ready and all(k.startswith("col:") or k in {"assignment", "change", "grain", "sampling"} for k in v["status"].open)
    v = say(g, cfg, "run")  # drafts confirmed → ready → written in the same turn
    assert v["written"] and v["handoff_ready"]


# ------------------------------------------------------------------ extract gate


def test_extract_gate_rejects_numbers_only_cites_and_bad_columns(tmp_path):
    csv = synthetic_cutoff(tmp_path)
    seen = []

    def dynamic(src, human):
        seen.append(human)
        if "REJECTED" in human:
            return cutoff_turn0()
        return [U("assignment", ["col:score.profile.numeric"], kind="cutoff_rule", rule="from the numbers", score_column="score", cutoff="0", treated_side="below"),
                U("measured", ["doc:context"], column="nope", meaning="x", when="before"),
                U("unobserved", ["doc:context"], exists="true")]

    fake = FakeLLM(dynamic=dynamic)
    g, cfg = start(fake, csv, "x")
    v = g.get_state(cfg).values
    assert len(seen) == 2 and "needs a doc or user cite" in seen[1] and "not a column" in seen[1] and "never read from a description" in seen[1]
    assert v["claims"].get("assignment").status == "drafted" and v["claims"].get("unobserved").status == "empty"


# ------------------------------------------------------------------ probes and the table


def test_thin_sides_strike_discontinuity_out(tmp_path):
    csv = synthetic_cutoff(tmp_path, n=30)
    fake = FakeLLM({"doc:context": cutoff_turn0()})
    g, cfg = start(fake, csv, "x")
    v = g.get_state(cfg).values
    pr = next(p for p in v["probes"] if p.family == "discontinuity")
    assert pr.passed is False and "rows_by_side" in v["status"].struck["discontinuity"]
    assert "unobserved" not in v["status"].required  # the struck families' own claims are never asked


def test_table_cells_and_flag():
    df = pd.DataFrame({"x": np.linspace(-1, 1, 100), "t": ([1] * 50) + ([0] * 50), "y": np.arange(100.0)})
    table = ClaimTable(claims={
        "grain": Claim(kind="grain", key="grain", fields={"row_is": "a unit", "panel": False}, status="confirmed", source="user:turn:1"),
        "sampling": Claim(kind="sampling", key="sampling", fields={"how": "whole"}, status="confirmed", source="user:turn:1"),
        "change": Claim(kind="change", key="change", fields={"what": "a grant", "to_whom": "units", "when": "2020"}, status="confirmed", source="user:turn:1"),
        "assignment": Claim(kind="assignment", key="assignment", fields={"kind": "cutoff_rule", "rule": "r", "score_column": "x", "cutoff": 0.0, "treated_side": "below"}, status="confirmed", source="user:turn:1"),
        "missing": Claim(kind="missing", key="missing", fields={"why": "none"}, status="confirmed", source="data"),
        "col:x": Claim(kind="measured", key="col:x", fields={"meaning": "score", "when": "before"}, status="confirmed", source="user:turn:1"),
        "col:y": Claim(kind="measured", key="col:y", fields={"meaning": "outcome", "when": "after"}, status="confirmed", source="user:turn:1"),
        "unobserved": Claim(kind="unobserved", key="unobserved"), "exclusion": Claim(kind="exclusion", key="exclusion"),
        "spillover": Claim(kind="spillover", key="spillover"), "trend_continues": Claim(kind="trend_continues", key="trend_continues"),
        "cutoff_only": Claim(kind="cutoff_only", key="cutoff_only"),
    })
    st = T.compute(CAT, table, [])
    assert st.table["diff_in_diff"]["grain"] == "does_not_fit" and st.table["diff_in_diff"]["assignment"] == "does_not_fit"
    assert st.table["discontinuity"]["assignment"] == "fits" and st.table["adjustment"]["unobserved"] == "unknown"
    assert st.table["discontinuity"]["unobserved"] == "not_needed" and st.table["discontinuity"]["cutoff_only"] == "unknown"
    assert set(st.surviving) == {"discontinuity", "adjustment", "instrument"}
    assert set(st.open) == {"unobserved", "spillover", "exclusion", "cutoff_only"} and not st.ready
    for k in ("unobserved", "spillover", "exclusion", "cutoff_only"):
        table.claims[k].status = "unknown"
    st = T.compute(CAT, table, [])
    assert st.ready
    st = T.compute(CAT, table, [ProbeResult(family="discontinuity", name="rows_by_side", passed=False, detail="thin")])
    assert "discontinuity" in st.struck and st.ready
    for f in list(CAT.families):
        table.claims["assignment"].fields["kind"] = "date_by_others"
        table.claims["grain"].fields["panel"] = False
    st = T.compute(CAT, table, [])
    assert set(st.surviving) == {"adjustment"}


# ------------------------------------------------------------------ catalogue and the written note


def test_catalogue_is_consistent_and_names_no_dataset():
    from causal_agent.intake.interview.checks import CHECKS

    names = set(yaml.safe_load((ROOT / "data/datasets.yaml").read_text()))
    for path in ("causal_agent/intake/knowledge/claims.yaml", "causal_agent/intake/knowledge/checks.yaml"):
        text = (ROOT / path).read_text().lower()
        for n in names:
            assert re.search(rf"\b{re.escape(n)}\b", text) is None, f"{path} names dataset {n}"
    for k in CAT.kinds.values():
        assert k.check == "none" or k.check in CHECKS, k.name
        assert k.fields and k.frame and k.about
        for fam in ("adjustment", "diff_in_diff", "discontinuity"):
            assert fam not in k.frame.lower() and fam.replace("_", " ") not in k.frame.lower()
    for fam in CAT.families.values():
        for path in fam.fits:
            kind, _, field = path.partition(".")
            assert kind in CAT.kinds and field in CAT.kinds[kind].fields, path
    th = load_thresholds()
    assert th["interview"]["max_asks"] >= 1


def test_rendered_note_round_trips(tmp_path):
    csv = synthetic_cutoff(tmp_path)
    fake = FakeLLM({"doc:context": cutoff_turn0()})
    g, cfg = start(fake, csv, "x", name="cut")
    v = g.get_state(cfg).values
    fake.script["user:turn:1"] = confirm_all(v["claims"], "user:turn:1")
    v = say(g, cfg, "yes")
    fake.script["user:turn:2"] = [U(k, ["user:turn:2"], unknown=True) for k in v["status"].open]
    v = say(g, cfg, "don't know")
    assert v["status"].ready
    v = say(g, cfg, "run")
    w = v["written"]
    pack = load_pack("cut", tmp_path / w["note"], tmp_path / w["profile"], tmp_path / w["claims"])
    assert pack.unnoted_columns == [] and pack.unprofiled_notes == [] and len(pack.changes) == 1
    assert "cutoff is 0" in pack.render_changes()
    assert w["csv"] == "cut.csv"  # already under the root, so not copied; a file elsewhere is copied to data/raw/<name>/
    outside = tmp_path.parent / "elsewhere.csv"
    outside.write_bytes(csv.read_bytes())
    w2 = W.write_dataset("cut2", str(outside), profile(outside), v["claims"], [], entity=None, time=None)
    assert (tmp_path / "data/raw/cut2/elsewhere.csv").exists() and w2["csv"] == "data/raw/cut2/elsewhere.csv"
    assert pack.resolve("probe:discontinuity.rows_by_side")
