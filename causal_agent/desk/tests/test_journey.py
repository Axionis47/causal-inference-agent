"""The desk graph with a scripted model: the question first, validated; one question per turn to ready; the run; the chat after.
The memory is held in this process; nothing is written under data/."""

from __future__ import annotations

import uuid

import pytest
from langgraph.types import Command

from causal_agent.common.llm import set_llm
from causal_agent.desk import graph as G
from causal_agent.desk import pipeline
from causal_agent.desk.contracts import AfterReply, FieldUpdate, Inference, NumberStated, RunRecord
from causal_agent.desk.tests.fakes import QUESTION, DeskFake, answer_ask
from causal_agent.memory import store
from causal_agent.memory.records import Memory

HELD: dict[str, Memory] = {}


@pytest.fixture(autouse=True)
def _held(monkeypatch, tmp_path):
    HELD.clear()

    def memory_for(name, root=None):
        if name not in HELD:
            HELD[name] = store.migrate(name, write=False)
        return HELD[name]

    def snapshot(memory, n, root=None):
        d = tmp_path / "designs" / str(n)
        d.mkdir(parents=True, exist_ok=True)
        (d / "memory.json").write_text(memory.model_dump_json())
        return d

    monkeypatch.setattr(store, "memory_for", memory_for)
    monkeypatch.setattr(store, "save", lambda m, root=None: None)
    monkeypatch.setattr(store, "snapshot", snapshot)
    monkeypatch.setattr(pipeline, "run", canned_run)
    yield
    set_llm(None)


def canned_run(path, n, dataset, question, decision=None, decision_record=""):
    sr = {"status": "done", "design": {"estimator": "linear_regression", "contrast": {"treated": "completed", "control": "none"}, "checks": {"results": []}},
          "estimates": [{"contrast": "completed_vs_none", "method": "linear_regression", "value": 5.6, "ci_low": 3.7, "ci_high": 7.5, "secondary": False, "error": None}],
          "refutations": [{"contrast": "completed_vs_none", "refuter": "placebo_treatment_refuter", "kind": "falsification", "new_effect": 0.05, "passed": True}],
          "interpretations": [], "feasibility": None}
    return RunRecord(index=n, dataset=dataset, question=question, family="adjustment", specialist="dowhy", status="done", effect=5.6, ci_low=3.7, ci_high=7.5,
                     estimator="linear_regression", decision=decision or {}, decision_record=decision_record, specialist_result=sr, design_dir=str(path.parent))


class Desk:
    def __init__(self, fake, dataset="students"):
        set_llm(fake)
        self.fake, self.g = fake, G.compile_local()
        self.cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        self.payload = self._drive({"dataset": dataset})

    def _drive(self, inp):
        payload = None
        for mode, chunk in self.g.stream(inp, self.cfg, stream_mode=["updates"]):
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value
        return payload

    def say(self, text):
        self.payload = self._drive(Command(resume=text))
        return self.payload

    @property
    def values(self):
        return self.g.get_state(self.cfg).values

    def to_ready(self, max_turns=6):
        turns = 0
        while self.payload and self.payload["kind"] == "ask" and not self.payload["ready"]:
            assert turns < max_turns, f"still asking after {turns} turns: {self.payload['text']}"
            self.say(answer_ask(self.payload))
            turns += 1
        return turns


# ------------------------------------------------------------------ the question first, validated


def test_the_first_turn_asks_the_question_and_refuses_one_the_file_cannot_answer():
    d = Desk(DeskFake())
    assert d.payload["kind"] == "question" and "What is the causal question" in d.payload["text"] and "lunch" in d.payload["text"]
    assert d.payload["ready"] is False and d.payload["ask"] is None
    p = d.say("How many students passed?")
    assert p["kind"] == "question" and "not yet a question I can answer" in p["text"] and "does not ask what a change did" in p["text"]
    p = d.say("Why do gender and lunch matter for math scores?")  # the frame reads the cause as the outcome: same column
    assert p["kind"] == "question" and "not yet" in p["text"]
    p = d.say(QUESTION)
    assert p["kind"] == "ask" and d.values["question"] == QUESTION and d.values["invalid"] is None
    assert HELD["students"].said[0].text == "How many students passed?"  # every turn is remembered, even a refused question


def test_students_reaches_ready_in_the_frame_plus_a_few_questions_then_runs():
    fake = DeskFake()
    d = Desk(fake)
    p = d.say(QUESTION)
    # the note was mined once into drafts; the first question confirms them in one go
    assert p["ask"]["kind"] == "confirm" and "claim:assignment.kind" in p["ask"]["addresses"] and fake.calls.count("Extraction") == 1
    assert HELD["students"].field("claim:assignment.kind").status == "drafted" and HELD["students"].field("claim:unobserved.exists") is None
    turns = d.to_ready(max_turns=6)
    assert turns <= 5 and d.payload["ready"] and "Say run" in d.payload["text"]
    # the ready moment: the design in the question's words, the evidence with addresses, the figure, the struck families with a reason each
    text = d.payload["text"]
    assert "Design: adjustment" in text and "[probe:adjustment.overlap]" in text and "[probe:adjustment.arms]" in text
    assert "You said" in text and "nothing hidden" in text and "Set aside: " in text and "diff_in_diff" in text and "discontinuity" in text
    fig = d.payload["figure"]
    assert fig and fig["id"].startswith("overlap_lunch") and fig["kind"] == "bars" and f"[{'figure:' + fig['id']}]" in text
    assert d.values["decision"].chosen == "adjustment" and d.values["convinced_version"] == HELD["students"].version
    m = HELD["students"]
    assert m.field("claim:assignment.kind").status == "confirmed" and m.field("claim:assignment.kind").source == "user:turn:2"
    assert m.field("claim:unobserved.exists").value is False and m.field("claim:unobserved.exists").said == "nothing hidden"
    asked = [t for t in fake.humans["Inference"]]
    assert all("(settles:" in h for h in asked)  # every inference saw the question it was answering
    p = d.say("run")
    assert p["kind"] == "after" and p["ready"] and "Run 1" in p["text"] and "[estimate:completed_vs_none.value]" in p["text"]
    runs = d.values["runs"]
    # the run's figures: the ready-moment overlap first, then the estimate against its falsifications; the brief shows the run's own
    assert [f["id"] for f in runs[0].figures][1] == "effect_completed_vs_none" and runs[0].figures[0]["id"].startswith("overlap_")
    assert p["figure"]["id"] == "effect_completed_vs_none" and (tmp_dir := runs[0].design_dir) and (__import__("pathlib").Path(tmp_dir) / "figures.json").exists()
    assert len(runs) == 1 and runs[0].effect == 5.6 and runs[0].family == "adjustment" and d.values["phase"] == "after"
    assert fake.calls.count("FamilyDecision") == 0 and fake.calls.count("Choice") == 0  # one family stood, one figure fit: both by code
    assert (d.values["design_dir"]) and d.values["handoff"].design_id == 1


# ------------------------------------------------------------------ the file talks back


def test_a_timing_answer_the_file_refutes_is_asked_again_then_stands_as_a_contradiction():
    fake = DeskFake()
    d = Desk(fake)
    d.say(QUESTION)
    d.say("yes, all right")  # the drafts stand, lunch fixed before the change
    m = HELD["students"]
    assert m.field("col:lunch.when").value == "before" and m.field("col:lunch.when").status == "confirmed"
    p = d.say("actually lunch was after the course")
    assert m.field("col:lunch.when").value == "after" and m.field("col:lunch.when").status == "refuted"
    assert p["ask"]["kind"] == "columns" and "lunch" in p["ask"]["addresses"][0] and "The file disagrees" in p["text"] and "check:col:lunch.when.depends_on_before" in p["text"]
    p = d.say("no, lunch was after, I am sure")
    assert m.field("col:lunch.when").status == "contradiction" and m.field("col:lunch.when").value == "after"
    assert "col:lunch.when" not in [a for a in (p["ask"] or {}).get("addresses", [])]  # a contradiction is settled: not asked a third time
    turns = d.to_ready()
    assert d.payload["ready"]
    d.say("run")
    assert "col:lunch.when" in d.values["handoff"].contradictions


def test_run_before_ready_takes_the_drafts_on_their_word_and_asks_the_rest():
    d = Desk(DeskFake())
    d.say(QUESTION)
    p = d.say("run")
    m = HELD["students"]
    assert m.field("claim:assignment.kind").status == "confirmed" and m.field("claim:assignment.kind").said == "run"
    assert p["kind"] == "ask" and "Before I can run" in p["text"] and "unobserved" in p["ask"]["addresses"][0]


# ------------------------------------------------------------------ the chat after


def test_after_the_run_a_revision_goes_back_through_the_gate_and_the_checks():
    after = [
        AfterReply(kind="answer", text="It raised math scores by 5.6 points; the figure shows the estimate against the placebo.", cites=["estimate:completed_vs_none.value"],
                   numbers=[NumberStated(address="estimate:completed_vs_none.value", value=5.6)], figure="figure:effect_completed_vs_none"),
        AfterReply(kind="revise", text="I will re-check with the offer depending on lunch only.", updates=[FieldUpdate(address="claim:assignment.depends_on", value="lunch", said="only lunch")]),
        AfterReply(kind="done", text="Bye."),
    ]
    fake = DeskFake(after=after)
    d = Desk(fake)
    d.say(QUESTION)
    d.to_ready()
    d.say("run")
    p = d.say("what did you find?")
    assert p["kind"] == "after" and "5.6" in p["text"] and p["figure"]["id"] == "effect_completed_vs_none"
    p = d.say("the offer only depended on lunch, not on parents")
    m = HELD["students"]
    assert m.value("claim:assignment.depends_on") == ["lunch"] and m.field("claim:assignment.depends_on").source.startswith("user:turn:")
    assert p["kind"] == "ask" and p["ready"] and d.values["phase"] == "before"  # everything still settled: back at the ready point
    p = d.say("run")
    assert p["kind"] == "after" and len(d.values["runs"]) == 2 and "Then and now" in p["text"]
    p = d.say("done")
    assert p is None


def test_a_what_if_runs_on_a_copy_and_leaves_the_memory_alone():
    after = [
        AfterReply(kind="what_if", text="Suppose places had been drawn by lot.", updates=[FieldUpdate(address="claim:assignment.kind", value="lottery", said="if it had been a lottery")]),
        AfterReply(kind="done", text="Bye."),
    ]
    d = Desk(DeskFake(after=after))
    d.say(QUESTION)
    d.to_ready()
    d.say("run")
    v_before = HELD["students"].version
    p = d.say("what if the places had been drawn by lot?")
    m = HELD["students"]
    assert m.value("claim:assignment.kind") == "own_choice" and m.version == v_before  # nothing known changed
    runs = d.values["runs"]
    assert len(runs) == 2 and runs[1].what_if == {"claim:assignment.kind": "lottery"} and runs[1].family == "adjustment"
    assert p["kind"] == "after" and "what-if" in p["text"] and "[claim:assignment.kind] = lottery" in p["text"] and "Then and now" in p["text"]
    assert "claim:assignment.kind" in runs[1].differs and d.values["fork"] is None
    assert d.values["handoff"].design_id == 2 and d.values["handoff"].assignment["kind"] == "lottery"


def test_a_new_question_about_a_different_change_asks_the_relative_fields_again():
    after = [AfterReply(kind="requestion", text="A new question.", question="Did a standard lunch raise math scores?")]

    def frame_lunch(msg, addrs, human):
        return None

    fake = DeskFake(after=after, infer=frame_lunch)
    d = Desk(fake)
    d.say(QUESTION)
    d.to_ready()
    d.say("run")
    m = HELD["students"]
    assert m.value("col:lunch.when") == "before" and m.value("col:lunch.meaning")
    # the scripted frame reads any question as the prep course; make it read lunch as the change for this one
    fr = fake.answer.__func__  # noqa: F841
    from causal_agent.common.contracts import Candidate
    from causal_agent.desk.tests import fakes as FK

    original = FK.students_frame

    def lunch_frame():
        f = original()
        f.cause_candidates = [Candidate(column="lunch", reason="the change asked about", cites=["col:lunch.note"])]
        return f

    FK.students_frame = lunch_frame
    try:
        p = d.say("Did a standard lunch raise math scores?")
    finally:
        FK.students_frame = original
    assert d.values["question"] == "Did a standard lunch raise math scores?" and d.values["phase"] == "before"
    assert m.value("col:lunch.when") is None and m.value("claim:assignment.kind") is None  # relative to the old change: gone
    assert m.value("col:lunch.meaning") and m.value("claim:grain.row_is")  # what a column is, and the grain, carry over
    assert m.value("claim:assignment.treatment_column") == "lunch" and "asked again" in p["text"]


def test_a_lane_that_asks_back_gets_its_answer_and_runs_again(monkeypatch):
    calls = []

    def asking_run(path, n, dataset, question, decision=None, decision_record=""):
        calls.append(n)
        if n == 1:
            return RunRecord(index=n, dataset=dataset, question=question, family="adjustment", specialist="dowhy", status="ask", design_dir=str(path.parent),
                             specialist_result={"status": "ask", "ask": {"address": "claim:mediator.exists", "question": "Is there a column the change altered, through which its whole effect runs?"},
                                                "feasibility": {"stage": "ask", "reason": "nothing identifies the effect while a hidden factor stands"}})
        return canned_run(path, n, dataset, question, decision, decision_record)

    monkeypatch.setattr(pipeline, "run", asking_run)

    def infer(msg, addrs, human):
        if "nothing hidden" in msg:  # the person says a hidden factor exists this time
            return Inference(updates=[FieldUpdate(address="claim:unobserved.exists", value="true", said=msg)])
        if msg == "no mediator":
            return Inference(updates=[FieldUpdate(address="claim:mediator.exists", value="false", said=msg)])
        return None

    d = Desk(DeskFake(infer=infer))
    d.say(QUESTION)
    d.to_ready()
    p = d.say("run")
    assert p["kind"] == "ask" and p["phase"] == "before" and p["ask"]["addresses"] == ["claim:mediator.exists"] and p["ask"]["options"] == ["yes", "no"]
    assert "stopped before estimating" in p["text"] and d.values["runs"][0].status == "ask"
    p = d.say("no mediator")
    assert HELD["students"].value("claim:mediator.exists") is False
    while p and p["kind"] == "ask" and not p["ready"]:
        p = d.say(answer_ask(p))
    assert p["ready"]
    p = d.say("run")
    assert p["kind"] == "after" and calls == [1, 2] and d.values["runs"][1].status == "done"


def test_a_lane_ask_with_options_and_evidence_reaches_the_page_and_is_asked_once(monkeypatch):
    """A lane's own options and evidence are shown; the same address asked by two runs in a row goes to the brief."""
    ask = {"address": "claim:trend_continues.believed", "question": "Is there a reason the groups would have moved differently?", "options": ["yes", "no"],
           "because": "[check:c.pre_trends] hard: the leads differ", "evidence": ["check:c.pre_trends"]}

    def asking_run(path, n, dataset, question, decision=None, decision_record=""):
        return RunRecord(index=n, dataset=dataset, question=question, family="adjustment", specialist="dowhy", status="ask", design_dir=str(path.parent),
                         specialist_result={"status": "ask", "ask": ask, "feasibility": {"stage": "ask", "reason": "the groups were already moving apart"}})

    monkeypatch.setattr(pipeline, "run", asking_run)

    def infer(msg, addrs, human):
        if msg == "yes there is":
            return Inference(updates=[FieldUpdate(address="claim:trend_continues.believed", value="true", said=msg)])
        return None

    d = Desk(DeskFake(infer=infer))
    d.say(QUESTION)
    d.to_ready()
    p = d.say("run")
    assert p["kind"] == "ask" and p["ask"]["addresses"] == ["claim:trend_continues.believed"] and p["ask"]["options"] == ["yes", "no"] and p["ask"]["evidence"] == ["check:c.pre_trends"]
    assert "the leads differ" in p["text"] and "moving apart" in p["text"]
    p = d.say("yes there is")
    while p and p["kind"] == "ask" and not p["ready"]:
        p = d.say(answer_ask(p))
    p = d.say("run")
    # the lane asked the same address again: the desk does not ask it twice, the brief says the lane still asks it
    assert p["kind"] == "after" and "still asks one thing" in p["text"] and "[ask.question]" in p["text"]
    assert [r.status for r in d.values["runs"]] == ["ask", "ask"]


def test_the_brief_lists_where_the_lane_disagreed_with_the_pack(monkeypatch):
    def declining_run(path, n, dataset, question, decision=None, decision_record=""):
        rec = canned_run(path, n, dataset, question, decision, decision_record)
        rec.specialist_result["declines"] = [{"stage": "load", "kind": "declined", "about": "scope.window", "pack_value": "the spring term", "reason": "the window is not in a form the code can apply; every row was kept", "check": "intake.window_unparsed"}]
        return rec

    monkeypatch.setattr(pipeline, "run", declining_run)
    d = Desk(DeskFake())
    d.say(QUESTION)
    d.to_ready()
    p = d.say("run")
    assert "disagreed with what was settled" in p["text"] and "[decline:load.scope_window]" in p["text"] and "every row was kept" in p["text"]
    from causal_agent.desk import material as M

    m = M.render(d.values["runs"][0])
    assert "decline:load.scope_window" in m.addresses and "pack said 'the spring term'" in m.by_address["decline:load.scope_window"]


def test_the_desk_shows_what_the_lane_drew_and_marks_the_ready_figure(monkeypatch, tmp_path):
    """A lane's own figures.json is what the run shows, after the ready-moment figure; without one the post-viz fallback draws."""
    import json

    def drawing_run(path, n, dataset, question, decision=None, decision_record=""):
        rec = canned_run(path, n, dataset, question, decision, decision_record)
        run_dir = tmp_path / "run"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "figures.json").write_text(json.dumps([
            {"id": "causal_graph", "kind": "graph", "title": "g", "series": [], "marks": [], "note": "", "draws_on": ["design.graph"],
             "nodes": [{"id": "a", "label": "a", "role": "treatment"}, {"id": "b", "label": "b", "role": "outcome"}], "edges": [{"src": "a", "dst": "b", "cites": []}]},
            {"id": "effect_completed_vs_none", "kind": "interval", "title": "e", "series": [{"name": "effect", "x": ["lr"], "y": [5.6]}], "marks": [], "note": "", "draws_on": []},
        ]))
        rec.run_dir = str(run_dir)
        return rec

    monkeypatch.setattr(pipeline, "run", drawing_run)
    d = Desk(DeskFake())
    d.say(QUESTION)
    d.to_ready()
    p = d.say("run")
    figs = d.values["runs"][0].figures
    assert [f["id"] for f in figs][1:] == ["causal_graph", "effect_completed_vs_none"] and figs[0]["moment"] == "ready" and figs[0]["id"].startswith("overlap_")
    assert p["figure"]["id"] == "causal_graph"
    from causal_agent.desk import material as M

    m = M.render(d.values["runs"][0])
    assert "figure:causal_graph.edge.0" in m.addresses and "(before the run)" in m.by_address[figs[0]["id"] and f"figure:{figs[0]['id']}"]
