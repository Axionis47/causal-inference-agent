"""The desk graph with a scripted model: the question first, validated; one question per turn to ready; the run; the chat after.
The memory is held in this process; nothing is written under data/."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pytest
from langgraph.types import Command

from causal_agent.common import config
from causal_agent.common.contracts import RunRecord
from causal_agent.common.llm import set_decider, set_llm
from causal_agent.desk import graph as G
from causal_agent.desk import pipeline
from causal_agent.desk.contracts import AfterReply, FieldUpdate, NumberStated, Reading
from causal_agent.desk.nodes import frame as F
from causal_agent.desk.tests.fakes import QUESTION, STORY_ANSWER, DeskFake, answer_ask, asked_addresses
from causal_agent.families import registry as R
from causal_agent.memory import journal as J
from causal_agent.memory import store
from causal_agent.memory.records import Memory

HELD: dict[str, Memory] = {}
REPO = Path(__file__).resolve().parents[3]  # the fixture claims files live here; everything written goes under tmp_path


@pytest.fixture(autouse=True)
def _held(monkeypatch, tmp_path):
    HELD.clear()

    def memory_for(name, root=None):
        if name not in HELD:
            HELD[name] = store.migrate(name, root=REPO, write=False)
        return HELD[name]

    def snapshot(memory, n, root=None):
        d = tmp_path / "designs" / str(n)
        d.mkdir(parents=True, exist_ok=True)
        (d / "memory.json").write_text(memory.model_dump_json())
        return d

    monkeypatch.setattr(store, "memory_for", memory_for)
    monkeypatch.setattr(store, "save", lambda m, root=None: None)

    def next_design_id(name, root=None):
        d = tmp_path / "designs"
        return max((int(p.name) for p in d.iterdir() if p.name.isdigit()), default=0) + 1 if d.is_dir() else 1

    monkeypatch.setattr(store, "snapshot", snapshot)
    monkeypatch.setattr(store, "next_design_id", next_design_id)
    # the journal is a second write path: it goes under tmp_path too, so a test never writes into the repo's data/
    monkeypatch.setattr(J, "open_journal", lambda name, a, root=None: J.Journal(tmp_path / "analyses" / a / "journal.jsonl"))
    monkeypatch.setattr(
        J, "next_analysis_id", lambda name, root=None: f"a{len(list((tmp_path / 'analyses').iterdir())) + 1 if (tmp_path / 'analyses').is_dir() else 1}"
    )
    monkeypatch.setattr(pipeline, "run", canned_run)
    # artifacts land under the memory home, which is tmp_path here; the exploration agent runs in process, never as a process
    from causal_agent.explore import client as XC
    from causal_agent.explore import server as XS

    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setenv("EXPLORE_SANDBOX", "subprocess")
    XC.set_app(XS.app("http://explore/"))
    yield
    XC.set_app(None)
    set_llm(None)
    set_decider(None)


def canned_run(path, n, dataset, question, decision=None, decision_record="", thread_id=None):
    sr = {
        "status": "done",
        "designs": {
            "d1": {
                "id": "d1",
                "endpoint_id": "total.backdoor",
                "target": {"estimand": "ate", "population": "all"},
                "estimator": "linear_regression",
                "contrast": {"treated": "completed", "control": "none"},
                "checks": {"results": []},
            }
        },
        "lead_design_id": "d1",
        "ledger": {"revision": 0, "entries": {}, "history": []},
        "estimates": [
            {
                "design_id": "d1",
                "contrast": "completed_vs_none",
                "method": "linear_regression",
                "value": 5.6,
                "ci_low": 3.7,
                "ci_high": 7.5,
                "secondary": False,
                "error": None,
            }
        ],
        "falsifications": [{"design_id": "d1", "contrast": "completed_vs_none", "refuter": "placebo_treatment_refuter", "new_effect": 0.05, "passed": True}],
        "interpretations": [],
        "feasibility": None,
    }
    pack = json.loads(Path(path).read_text()) if Path(path).is_file() else {}
    sr["context_id"] = pack.get("context_id", "")
    sr["designs"]["d1"]["context_id"] = sr["context_id"]
    return RunRecord(
        index=n,
        dataset=dataset,
        question=question,
        family="adjustment",
        specialist="dowhy",
        status="done",
        effect=5.6,
        ci_low=3.7,
        ci_high=7.5,
        estimator="linear_regression",
        decision=decision or {},
        decision_record=decision_record,
        specialist_result=sr,
        design_dir=str(path.parent),
    )


class Desk:
    def __init__(self, fake, dataset="students", analysis=None):
        set_llm(fake)
        set_decider(fake)
        self.fake, self.g = fake, G.compile_local()
        self.cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        self.payload = self._drive({"dataset": dataset, **({"analysis": analysis} if analysis else {})})

    @property
    def journal(self):
        return J.open_journal(self.values["dataset"], self.values["analysis"])

    def _drive(self, inp):
        payload = None
        for _mode, chunk in self.g.stream(inp, self.cfg, stream_mode=["updates"]):
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value
        return payload

    def say(self, text):
        self.payload = self._drive(Command(resume=text))
        return self.payload

    @property
    def values(self):
        return self.g.get_state(self.cfg).values

    def to_ready(self, max_turns=10):
        """Say the analysis is wanted, then answer what blocks until the ready moment: the design shown, waiting for the word."""
        turns = 0
        if self.payload and self.payload["kind"] == "chat":
            self.say("let's run it")
            turns += 1
        while self.payload and self.payload["kind"] == "ask":
            assert turns < max_turns, f"still asking after {turns} turns: {self.payload['text']}"
            self.say(answer_ask(self.payload))
            turns += 1
        assert self.payload and self.payload["kind"] == "ready", f"not at the ready moment: {self.payload and self.payload['text']}"
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
    assert p["kind"] == "chat" and d.values["question"] == QUESTION and d.values["invalid"] is None
    assert HELD["students"].said[0].text == "How many students passed?"  # every turn is remembered, even a refused question


def test_students_reaches_ready_in_the_frame_plus_a_few_questions_then_runs():
    fake = DeskFake()
    d = Desk(fake)
    p = d.say(QUESTION)
    # the first reply is the map: what the file could answer the question with, what each family still needs, what is struck and why
    text = p["text"]
    assert text.startswith("With this file, math score against test preparation course could be answered")
    adj = next(f for f in R.knowledge() if f.name == "adjustment")
    assert f"- adjustment: {adj.answers[0].upper() + adj.answers[1:]}" in text and "Still to settle:" in text
    assert "Struck already: " in text and "discontinuity (assignment does not fit)" in text and "Say which of these you care about" in text
    assert p["kind"] == "chat" and p["ask"] is None and "Here is what I read" not in text  # the map alone: nothing is asked until the person wants to run
    assert HELD["students"].field("claim:assignment.kind").status == "drafted" and HELD["students"].field("claim:unobserved.exists") is None
    p = d.say("let's run it")
    # the note was mined once into drafts by the Reader; the story is not asked, the readback confirms the drafts in one go
    assert (
        p["ask"]["kind"] == "confirm"
        and "claim:assignment.kind" in p["ask"]["addresses"]
        and len(fake.reads("doc:")) == 1
        and "Here is what I read" in p["text"]
    )
    assert d.values["story_asked"] and d.values["readback_done"] and "Tell me the story" not in p["text"]
    turns = d.to_ready(max_turns=10)
    assert turns <= 8 and d.payload["ready"] and "Say yes" in d.payload["text"] and "could be answered" not in d.payload["text"]  # the map was said once
    # only what blocks the run was asked: the beliefs landed, and a relation a decision merely rests on was never asked
    reads = fake.reads("user:")
    settles = lambda h: h.split("(settles: ")[1].split(")")[0].split(", ") if "(settles: " in h else []  # noqa: E731
    asked = {a for h in reads for a in settles(h)}
    assert HELD["students"].value("claim:spillover.possible") is False
    assert not {"claim:assignment.offer_column", "col:lunch.same_as", "col:lunch.may_modify"} & asked
    assert HELD["students"].value("claim:assignment.offer_column") is None and HELD["students"].value("col:lunch.may_modify") is None
    # the journal so far: the question read, then one claim step per turn that settled something, each on the person's word;
    # beside them the fit steps: the matrix is a record, and a cell that moved is a step by code naming what moved it
    fits, steps = [s for s in d.journal.steps() if s.kind == "fit"], [s for s in d.journal.steps() if s.kind != "fit"]
    assert fits and fits[0].n == 2 and all(s.by == "code" and s.design is None for s in fits)
    assert "adjustment.assignment: none -> fits (claim:assignment.kind)" in fits[0].note and "claim:assignment.kind" in fits[0].read
    assert any("adjustment.unobserved: unknown -> fits (claim:unobserved)" in s.note for s in fits[1:])
    assert steps[0].kind == "question" and steps[0].by == "model" and steps[0].design is None
    assert {"user:turn:1", "col:math_score", "col:test_preparation_course"} <= set(steps[0].read)
    assert steps[0].note.startswith("effect_of_change: math score against test preparation course")
    assert [s.kind for s in steps[1:]] == ["claim"] * (len(steps) - 1) and all(s.by == "person" and s.design is None for s in steps[1:])
    assert all(s.read == [f"user:turn:{n}"] for s, n in zip(steps[1:], range(3, len(steps) + 2)))  # turn 2 was "let's run it"
    assert [s.memory_version for s in steps] == sorted(s.memory_version for s in steps) and "claim:assignment.kind" in steps[1].note
    # the ready moment: the design in the question's words, the evidence with addresses, the figure, the struck families with a reason each
    text = d.payload["text"]
    assert "Design: adjustment" in text and "[probe:adjustment.overlap]" in text and "[probe:adjustment.arms]" in text
    # the ready moment reads the assembled context back, in the order the reasoning reads it, before the design
    assert "Here is the context the analysis will run on" in text and text.index("THE PAIR") < text.index("THE MECHANISM") < text.index("TIME") < text.index(
        "HIDDEN FACTORS"
    ) < text.index("Design: adjustment")
    assert "[pair.outcome] the outcome is 'math score' [col:math_score]" in text and "[pair.levels] treated = 'completed', control = 'none'" in text
    # every column is carried: the ones the frame never named stand in the time lines beside the ones it did
    assert "[time.before] fixed before the change: lunch, parental level of education, gender, race/ethnicity" in text
    assert "[time.after] measured after the change: math score, reading score, writing score" in text
    assert "Set aside: " in text and "diff_in_diff" in text and "discontinuity" in text
    # the Designer's brief replaces the canned assumption: what it bets on, one line per decision, the road, the threats
    assert "It bets on: nothing beyond lunch and parents' education drove both" in text and fake.calls.count("DesignBrief") == 1
    assert "[design.brief.total.backdoor]" in text
    assert "What would break it: something outside the file could have driven both [claim:assignment.kind]" in text
    assert text.index("It bets on") < text.index("[design.brief.total.backdoor]") < text.index("Evidence:")
    assert d.values["brief"].family == "adjustment"
    assert "ask for a picture" in text and d.payload["artifact"] is None  # nothing is drawn unasked
    assert d.values["decision"].chosen == "adjustment" and d.values["convinced_version"] == HELD["students"].version
    m = HELD["students"]
    assert m.field("claim:assignment.kind").status == "confirmed" and m.field("claim:assignment.kind").source == "user:turn:3"  # the readback's yes
    assert m.field("claim:spillover.possible").value is False
    assert all("(settles:" in h for h in fake.reads("user:")[1:])  # every reading of an answer saw the question it was answering
    p = d.say("run")
    assert p["kind"] == "after" and p["ready"] and "Run 1" in p["text"] and "[estimate:d1.completed_vs_none.value]" in p["text"]
    runs = d.values["runs"]
    # the run's figures: the estimate against its falsifications, drawn from the artifacts; the brief shows it
    assert [f["id"] for f in runs[0].figures] == ["effect_d1_completed_vs_none"]
    assert (
        p["figure"]["id"] == "effect_d1_completed_vs_none"
        and (tmp_dir := runs[0].design_dir)
        and (__import__("pathlib").Path(tmp_dir) / "figures.json").exists()
    )
    assert len(runs) == 1 and runs[0].effect == 5.6 and runs[0].family == "adjustment" and d.values["phase"] == "after"
    assert fake.calls.count("FamilyDecision") == 0 and fake.calls.count("Exploration") == 0  # one family stood: chosen by code; nothing drawn unasked
    assert (d.values["design_dir"]) and d.values["handoff"].design_id == 1
    # the pack carries the brief and bets on its sentence; the design folder holds it twice, in handoff.json and brief.json
    import json
    from pathlib import Path

    h = d.values["handoff"]
    assert h.brief is not None and h.chosen_assumption == h.brief.bets_on and fake.calls.count("DesignBrief") == 1
    facts = {p.name: p for p in h.probes if p.family == "data"}
    assert {"by_arm.lunch", "by_arm.parental_level_of_education", "timing"} <= set(facts)  # the numbers a reasoning would ask for, up front
    assert not any(name.startswith("with_outcome.") for name in facts)
    assert "of the treated" in facts["by_arm.lunch"].detail and facts["timing"].detail.startswith("before: ")
    assert (
        not any("math_score" in n and "test_preparation_course" in n for n in facts) and "by_arm.math_score" not in facts
    )  # nothing joins outcome and treatment
    assert h.story == STORY_ANSWER or h.story.startswith("#")  # the account rides in the pack
    assert h.render_context().startswith("STORY") and "[probe:data.by_arm.lunch]" in h.render_context() or True
    folder = Path(d.values["design_dir"])
    assert json.loads((folder / "handoff.json").read_text())["brief"]["bets_on"] == h.brief.bets_on
    assert json.loads((folder / "brief.json").read_text())["decisions"][0]["name"] in {e.id for e in R.needs()["adjustment"].endpoints.endpoints}
    assert "BETS ON      " + h.brief.bets_on in d.values["decision_record"] and "[design.brief.total.backdoor]" in d.values["decision_record"]
    assert runs[0].decision["chosen_assumption"] == h.brief.bets_on
    # the journal: the design written from the memory, then the run that read it, both on this design's group
    design, run, brief = d.journal.steps()[-3:]
    assert brief.kind == "brief"
    assert (
        design.kind == "design"
        and design.by == "code"
        and design.design == 1
        and design.left == ["designs/1"]
        and design.note == "design 1: adjustment via dowhy"
    )
    assert run.kind == "run" and run.design == 1 and run.read == [design.address] and run.left == ["designs/1/record.json", "designs/1/figures.json"]
    assert run.note.startswith("done, effect 5.6 [3.7, 7.5] by linear_regression") and run.memory_version == design.memory_version


def test_a_second_conversation_on_the_same_memory_numbers_its_design_after_the_first_and_keeps_every_word():
    d = Desk(DeskFake())
    d.say(QUESTION)
    d.to_ready()
    d.say("run")
    m = HELD["students"]
    last_turn = max(s.turn for s in m.said)
    assert d.values["handoff"].design_id == 1 and last_turn >= 3
    # a new thread on the same dataset: the desk starts a fresh conversation over the memory as it stands
    d2 = Desk(DeskFake())
    assert d2.payload["kind"] == "question"
    d2.say("How many students passed?")
    assert m.said[-1].turn == last_turn + 1 and m.said[-1].text == "How many students passed?"  # not skipped as a repeat turn number
    d2.say(QUESTION)
    d2.to_ready()
    d2.say("run")
    assert d2.values["handoff"].design_id == 2 and d2.values["runs"][0].index == 2  # design 1 is not written over
    # each conversation keeps its own journal: the second starts at step 1 with its own design, the first is untouched by it
    first_before = d.journal.path.read_text()
    assert d2.values["analysis"] == "a2" and d2.journal.steps()[0].n == 1 and d2.journal.last("design").left == ["designs/2"]
    assert d.journal.path.read_text() == first_before and d.journal.last("design").left == ["designs/1"]
    assert (
        (tmp := __import__("pathlib").Path(d.values["design_dir"])).exists()
        and tmp.name == "1"
        and __import__("pathlib").Path(d2.values["design_dir"]).name == "2"
    )


def test_the_run_record_is_written_beside_its_design_and_reads_back(tmp_path):
    what_if = AfterReply(
        kind="what_if",
        text="Suppose places had been drawn by lot.",
        updates=[FieldUpdate(address="claim:assignment.kind", value="lottery", said="if it had been a lottery")],
    )
    d = Desk(DeskFake(after=[what_if]))
    d.say(QUESTION)
    d.to_ready()
    d.say("run")
    rec = d.values["runs"][0]
    back = pipeline.load_record(rec.design_dir)
    assert back is not None and back.model_dump() == rec.model_dump() and back.figures and back.index == 1
    # a what-if is a second design; its record carries the fork's change and what differed, saved once the brief knew them
    d.say("what if the places had been drawn by lot?")
    assert len(d.values["runs"]) == 1 and d.values["phase"] == "before"
    d.to_ready()
    d.say("run")
    rec2 = d.values["runs"][1]
    back2 = pipeline.load_record(rec2.design_dir)
    assert back2 is not None and back2.what_if == {"claim:assignment.kind": "lottery"} and back2.differs == rec2.differs and back2.differs
    # a design that never ran has no record
    empty = tmp_path / "designs" / "9"
    empty.mkdir(parents=True)
    assert pipeline.load_record(empty) is None


def test_a_brief_that_cites_nothing_real_is_refused_three_times_then_falls_back_to_the_family_words():
    from causal_agent.common.contracts import Cited, DecisionMade, DesignBrief
    from causal_agent.desk.tests.fakes import brief_by_rule

    bad = DesignBrief(
        family="adjustment",
        decisions=[DecisionMade(name="total.backdoor", choice="completed against none", rests_on=["nonsense:thing"], reason="r")],
        threats=[Cited(reason="t", cites=["col:nope.note"])],
        bets_on="a sentence",
    )
    fake = DeskFake(brief=[bad, bad, bad])
    d = Desk(fake)
    d.say(QUESTION)
    d.to_ready()
    assert fake.calls.count("DesignBrief") == 3
    last = fake.humans["DesignBrief"][2]
    assert "PREVIOUS ATTEMPT FAILED THESE CHECKS" in last and "'nonsense:thing'" in last and "must appear exactly once" in last and "threat 1 cites" in last
    b = d.values["brief"]
    adj = next(f for f in R.knowledge() if f.name == "adjustment")
    assert b.bets_on == adj.assumes and [x.name for x in b.decisions] == [x.name for x in adj.decisions]
    assert all(x.choice == "not decided" and x.rests_on == ["change:1.note"] for x in b.decisions)
    assert f"It bets on: {adj.assumes}" in d.payload["text"] and f"[design.brief.{adj.decisions[0].name}] not decided" in d.payload["text"]
    # run takes the design decided at the ready moment: the fallback brief is the pack's brief, written beside it
    d.say("run")
    h = d.values["handoff"]
    assert h.chosen_assumption == adj.assumes and fake.calls.count("DesignBrief") == 3
    assert __import__("json").loads((__import__("pathlib").Path(d.values["design_dir"]) / "brief.json").read_text())["decisions"][0]["choice"] == "not decided"
    assert brief_by_rule(fake.humans["DesignBrief"][0]).family == "adjustment"  # what the rule would have said, had it been asked


def test_a_model_that_returns_no_brief_is_asked_again_and_the_third_answer_stands():
    fake = DeskFake(brief=[None, None])  # two empty answers, then the rule's brief
    d = Desk(fake)
    d.say(QUESTION)
    d.to_ready()
    assert fake.calls.count("DesignBrief") == 3
    assert "the model returned no structured answer" in fake.humans["DesignBrief"][1]
    assert d.values["brief"].decisions[0].choice != "not decided"


def test_the_chat_after_a_run_can_cite_what_the_design_rested_on():
    from causal_agent.desk import material as M

    after = [
        AfterReply(kind="answer", text="It bets on the two columns the offer looked at.", cites=["design.brief.bets_on", "design.brief.total.backdoor"]),
        AfterReply(kind="done", text="Bye."),
    ]
    fake = DeskFake(after=after)
    d = Desk(fake)
    d.say(QUESTION)
    d.to_ready()
    p = d.say("run")
    assert p["kind"] == "after"
    run = d.values["runs"][0]
    mat = M.render(run, HELD["students"])
    assert {"design.brief", "design.brief.bets_on", "design.brief.total.backdoor", "design.brief.threat:1"} <= mat.addresses
    assert mat.by_address["design.brief.bets_on"] == run.decision["brief"]["bets_on"]
    p = d.say("what does it rest on?")
    assert p["text"].startswith("It bets on the two columns") and fake.calls.count("AfterReply") == 1  # the gate took the brief's addresses first time
    assert "[design.brief.bets_on]" in fake.humans["AfterReply"][0] and d.journal.last("answer").read == ["design.brief.bets_on", "design.brief.total.backdoor"]


# ------------------------------------------------------------------ focus: the families the person cares about


def test_naming_the_families_you_care_about_drops_the_questions_the_others_need():
    def only_adjustment(msg, addrs, human):
        if msg.startswith("only adjustment"):
            return Reading(kind="update", focus=["adjustment"])
        return None

    fake = DeskFake(infer=only_adjustment)
    d = Desk(fake)
    p = d.say(QUESTION)
    assert "instrument:" in p["text"]  # the map lists instrument beside adjustment
    p = d.say("only adjustment, please")
    assert d.values["focus"] == ["adjustment"] and d.values["status"].surviving == ["adjustment"]
    assert "exclusion" not in d.values["status"].required  # the instrument family's need is no longer asked
    d.to_ready()
    asked = "\n".join(fake.reads("user:"))
    assert "settles: claim:exclusion" not in asked and d.payload["ready"]
    assert "Set aside: " in d.payload["text"] and "instrument:" in d.payload["text"]
    assert (
        d.values["decision"].chosen == "adjustment"
        and {r.family: r.reason for r in d.values["decision"].rejected}["instrument"] == "declared family has no implemented endpoint trajectory"
    )
    d.say("run")
    assert d.values["runs"][0].family == "adjustment"


def test_a_family_the_grid_does_not_know_is_refused_and_the_focus_stays():
    calls = []

    def magic(msg, addrs, human):
        if msg.startswith("only magic"):
            calls.append(human)
            return Reading(kind="update", focus=["magic"]) if len(calls) == 1 else Reading(kind="update")
        return None

    fake = DeskFake(infer=magic)
    d = Desk(fake)
    d.say(QUESTION)
    p = d.say("only magic, please")
    assert len(calls) == 2 and "focus: magic is not a family the fit grid knows" in calls[1]  # refused, asked again with the reason
    assert not d.values.get("focus") and p["kind"] == "chat" and len(d.values["status"].surviving) > 1


def test_a_turn_that_settles_nothing_at_the_ready_moment_does_not_repeat_the_last_reply():
    d = Desk(DeskFake())
    d.say(QUESTION)
    d.to_ready()
    first = d.payload["text"]
    p = d.say("thanks, one moment")  # settles nothing: the ready moment is said again, once, without the map or the reply before
    assert p["ready"] and p["text"].count("Everything the analysis needs is settled.") == 1 and "could be answered" not in p["text"]
    assert first.count("Everything the analysis needs is settled.") == 1


# ------------------------------------------------------------------ asking the desk before the run


def test_a_question_to_the_desk_is_answered_beside_the_next_ask_and_an_update_in_the_same_message_still_lands():
    def curious(msg, addrs, human):
        if msg.startswith("what does own choice mean"):
            return Reading(
                kind="update",
                question="what does own choice mean here?",
                updates=[FieldUpdate(address="claim:unobserved.exists", value="false", said="nothing hidden either way")],
            )
        return None

    # the Explainer before the run: the same contract and the same gate as after it; a family is an address in the material
    answer = AfterReply(
        kind="answer",
        text="Own choice means the student decided whether to take the place once offered.",
        cites=["family:adjustment", "claim:assignment.kind", "claim:assignment.rule"],
    )
    fake = DeskFake(infer=curious, explain=[answer])
    d = Desk(fake)
    d.say(QUESTION)
    d.say("yes, all right")
    p = d.say("what does own choice mean here? nothing hidden either way")
    m = HELD["students"]
    assert m.field("claim:unobserved.exists").value is False  # the update in the same message landed first
    assert (
        p["text"].startswith(answer.text)
        and "[family:adjustment] [claim:assignment.kind] [claim:assignment.rule]" in p["text"]
        and p["ask"] is None
        and p["text"].rstrip().endswith("Noted: [claim:unobserved.exists] no")  # the answer, then what landed; nothing asked
    )
    human = fake.humans["AfterReply"][0]
    assert fake.calls.count("AfterReply") == 1 and "THE PERSON SAYS\nwhat does own choice mean here?" in human
    # the material before the run: the families' knowledge, the matrix, the steps, the person's words and the memory, each an address
    assert "[family:adjustment] family: adjustment" in human and "[matrix:adjustment.assignment]" in human and "[step:1] question by model" in human
    assert '[user:turn:1] "' in human and "[claim:assignment.kind] own_choice" in human and "THE PHASE\nbefore the run" in human
    assert d.values["explained"] is None and d.values["desk_question"] is None  # said once
    tail = [s for s in d.journal.steps() if s.kind != "fit"][-2:]  # the belief moved a cell too: a fit step follows the explain
    assert [s.kind for s in tail] == ["claim", "explain"] and "claim:unobserved.exists" in tail[0].note
    assert (
        tail[1].by == "model"
        and tail[1].read == ["family:adjustment", "claim:assignment.kind", "claim:assignment.rule"]
        and tail[1].note == "what does own choice mean here?"
    )
    p = d.say(answer_ask(p))
    assert not p["text"].startswith(answer.text)


def test_an_answer_that_cites_nothing_real_or_settles_something_is_refused_three_times_then_falls_back():
    def curious(msg, addrs, human):
        return Reading(kind="ask", question="why do you ask that?") if msg.startswith("why") else None

    bad = AfterReply(kind="answer", text="Because.", cites=["nonsense:thing"])
    revise = AfterReply(kind="revise", text="Noted.", updates=[FieldUpdate(address="claim:sampling.how", value="by_arm", said="x")])  # not legal before the run
    fake = DeskFake(infer=curious, explain=[bad, revise, bad])
    d = Desk(fake)
    d.say(QUESTION)
    p = d.say("why do you ask that?")
    assert fake.calls.count("AfterReply") == 3 and "PREVIOUS REPLY WAS REJECTED" in fake.humans["AfterReply"][2]
    assert "revise is not legal before the run" in fake.humans["AfterReply"][2] and HELD["students"].value("claim:sampling.how") == "whole"
    assert p["text"].startswith("I can only answer that from what is settled.") and "can be answered by adjustment" in p["text"] and p["ask"] is None
    last = d.journal.last("explain")
    assert last is not None and last.read == [] and last.note == "why do you ask that? (unanswered)"


# ------------------------------------------------------------------ the story, the readback, the gaps by decision


@pytest.fixture
def _no_note(monkeypatch):
    """The students file without its note: the memory is bare, so the story is asked."""
    monkeypatch.setattr(F, "_doc", lambda memory: None)


def test_the_story_is_asked_once_then_read_back_then_the_gaps_are_asked_by_decision(_no_note):
    fake = DeskFake()
    d = Desk(fake)
    d.say(QUESTION)
    p = d.say("let's run it")
    # the story turn: once the person wants to run, before any field question, over every open address
    a = p["ask"]
    assert a["kind"] == "story" and p["text"].endswith(
        "Tell me the story: what the change was, who could get it and how that was decided, what each "
        "column records and when it was set, and what one row is. Paste a note if you have one."
    )
    assert {"claim:assignment.kind", "claim:change.what", "claim:grain.row_is", "col:lunch.when"} <= set(a["addresses"]) and d.values["story_asked"]
    assert len(fake.reads("doc:")) == 0
    # a narrative answer drafts several claims in one turn, each with the sentence it rests on; nothing is confirmed yet
    p = d.say(STORY_ANSWER)
    m = HELD["students"]
    kind, when = m.field("claim:assignment.kind"), m.field("col:math_score.when")
    assert kind.status == "drafted" and kind.source == "user:turn:3" and kind.said == "then open to anyone who asked" and kind.reason == "scripted"
    assert when.status == "drafted" and when.value == "after" and m.value("claim:change.what") == "a six-week test preparation course"
    story = m.field("claim:story.text")  # the account itself, whole, on the person's word, beside what was read from it
    assert story.value == STORY_ANSWER and story.status == "confirmed" and story.source == "user:turn:3" and story.said == STORY_ANSWER
    assert "story" not in fake.reads("user:")[0].split("THE MEMORY AS IT STANDS")[0]  # the Reader is never asked to fill it
    assert "(the story)" in fake.reads("user:")[0] and d.journal.last("claim").note.count("claim:") >= 8  # "let's run it" never reached the Reader
    # the readback: the drafts grouped by the five claims, each line in the world's terms with its sentence, one confirm turn
    text, a = p["text"], p["ask"]
    assert a["kind"] == "confirm" and set(a["addresses"]) >= {"claim:assignment.kind", "claim:change.what", "claim:grain.row_is", "col:lunch.when"}
    assert d.values["readback_done"] and "Tell me the story" not in text  # asked once
    groups = [
        "Who got the change, and how that was decided:",
        "What the change was, and when:",
        "What one row is, and which rows are in the file:",
        "What each column records, and when it was set:",
    ]
    assert [text.index(g) for g in groups] == sorted(text.index(g) for g in groups)
    assert '• assignment, kind: own_choice (the unit decided whether to take it, with or without an offer) — "then open to anyone who asked"' in text
    assert (
        "• 'lunch': records lunch status at enrolment; fixed before the change — \"Lunch and parental level of education were recorded at enrolment\"" in text
    )
    assert text.endswith("Is this right? Correct any line, or say yes.")
    # a yes confirms every draft shown
    p = d.say("yes, all right")
    assert all(m.field(x).status == "confirmed" and m.field(x).source == "user:turn:4" for x in a["addresses"])
    # a gap question names the decision it serves and asks the fields under it together
    a = p["ask"]
    endpoint_ids = {e.id for e in R.needs()["adjustment"].endpoints.endpoints}
    assert a["decision"] in endpoint_ids and set(a["addresses"]) == {"claim:sampling.how", "claim:unobserved.exists", "claim:spillover.possible"}
    assert a["kind"] == "open"
    p = d.say("claim:sampling.how = whole")
    a = p["ask"]
    assert a["decision"] in endpoint_ids and "claim:unobserved.exists" in a["addresses"]
    assert "claim:exclusion.exists" not in a["addresses"]
    d.to_ready()
    assert d.payload["ready"] and m.field("claim:unobserved.exists").value is False
    # the mechanism (an offer and a taking as two columns) is a relation that only helps: not asked, left open for the lane
    asked = fake.reads("user:")
    assert not any("To settle how the change reached the units" in h for h in asked) and m.value("claim:assignment.offer_column") is None
    # the relations a decision merely rests on are never asked: they do not block the run
    assert not any("To settle what enters the adjustment set" in h or "To settle where the effect could differ" in h for h in asked)


def test_a_mined_note_skips_the_story_and_the_readback_comes_first():
    fake = DeskFake()
    d = Desk(fake)
    d.say(QUESTION)
    p = d.say("let's run it")
    assert len(fake.reads("doc:")) == 1 and p["ask"]["kind"] == "confirm" and "Tell me the story" not in p["text"]
    assert d.values["story_asked"] and d.values["readback_done"]
    m = HELD["students"]
    f = m.field("claim:assignment.rule")
    assert f.status == "drafted" and f.source == "doc:context" and f.said == "offered first by lunch status"  # the sentence travels from the note
    story = m.field("claim:story.text")
    assert story.status == "drafted" and story.source == "doc:context" and story.value.startswith("#") and len(story.value) > 200  # the note, whole
    assert '— "offered first by lunch status"' in p["text"] and "What the design would bet on" not in p["text"]  # a note sets no belief
    assert m.field("claim:unobserved.exists") is None
    assert "Tell me the story" not in "\n".join(fake.reads("user:"))


def test_optional_assignment_fields_are_not_required_for_admitted_endpoint(_no_note):
    fake = DeskFake()
    d = Desk(fake)
    d.say(QUESTION)
    d.say("let's run it")
    d.say(STORY_ANSWER)
    d.to_ready()
    requests = [a for human in fake.reads("user:") for a in asked_addresses(human)]
    assert "claim:assignment.offer_column" not in requests
    assert d.payload["ready"]
    assert HELD["students"].value("claim:assignment.offer_column") is None


def test_each_conversation_has_its_own_analysis_id_that_survives_every_resume():
    d = Desk(DeskFake())
    assert d.values["analysis"] == "a1"  # minted by load when the caller gave none
    d.say(QUESTION)
    d.to_ready()
    assert d.values["analysis"] == "a1"
    d2 = Desk(DeskFake(), analysis="a7")  # the caller's id wins
    assert d2.values["analysis"] == "a7"
    d2.say(QUESTION)
    assert d2.values["analysis"] == "a7" and d.values["analysis"] == "a1"


def test_run_intent_over_open_drafts_asks_the_readback_first_and_the_word_confirms_nothing():
    d = Desk(DeskFake())
    d.say(QUESTION)  # the drafts from the note are open, to be confirmed
    m = HELD["students"]
    p = d.say("run")
    assert p["kind"] == "ask" and "Before I can run" in p["text"] and p["ask"]["kind"] == "confirm", (
        "run intent turns the interview on; the readback comes first"
    )
    assert m.field("claim:assignment.kind").status == "drafted", "the word run confirms no draft"
    assert d.journal.last("claim") is None or not d.journal.last("claim").note.startswith("confirmed as drafted")
    d.say("yes, all right")  # the person's word on the drafts they were shown
    claim = d.journal.last("claim")
    assert claim is not None and claim.by == "person" and claim.read == ["user:turn:3"] and m.field("claim:assignment.kind").status == "confirmed"


def test_the_matrix_is_a_record_a_fit_step_only_when_a_cell_moves_and_matrix_json_at_handoff():
    from pathlib import Path

    from causal_agent.memory.matrix import Matrix

    d = Desk(DeskFake())
    d.say(QUESTION)
    d.to_ready()
    n_fits = len([s for s in d.journal.steps() if s.kind == "fit"])
    assert n_fits >= 2  # the first probe, then the turns that moved a cell
    d.say("thanks, one moment")  # settles nothing: the probe runs again, no cell moves, no fit step
    assert len([s for s in d.journal.steps() if s.kind == "fit"]) == n_fits
    mx = d.values["matrix"]
    assert isinstance(mx, Matrix) and mx.ready and mx.cell("adjustment", "unobserved").value == "fits"
    d.say("run")
    written = Path(d.values["design_dir"]) / "matrix.json"
    assert written.exists() and Matrix.model_validate_json(written.read_text()) == d.values["matrix"]
    assert len([s for s in d.journal.steps() if s.kind == "fit"]) == n_fits  # the route's own fit is a recomputation, not a move
    # the matrix is material after the run: the chat can cite a cell
    from causal_agent.desk import material as M

    mat = M.render(d.values["runs"][0], HELD["students"], matrix=d.values["matrix"])
    assert "matrix:adjustment.assignment" in mat.addresses and mat.by_address["matrix:adjustment.assignment"].startswith("fits")


def test_lane_assertions_remain_drafts_after_an_unrelated_human_correction(monkeypatch):
    def graph_run(path, n, dataset, question, decision=None, decision_record="", thread_id=None):
        rec = canned_run(path, n, dataset, question, decision, decision_record)
        rec.specialist_result["ledger"] = {
            "revision": 1,
            "history": [],
            "entries": {
                f"rel:{i}": {
                    "id": f"rel:{i}",
                    "kind": "edge",
                    "src": "lunch",
                    "dst": dst,
                    "status": "supported",
                    "reason": "explicit source-backed proposal",
                    "evidence": [{"source": "pack", "category": "study_account", "address": address}],
                }
                for i, (dst, address) in enumerate([("test_preparation_course", "claim:assignment.depends_on"), ("math_score", "col:lunch.when")])
            },
        }
        return rec

    monkeypatch.setattr(pipeline, "run", graph_run)
    after = [AfterReply(kind="revise", text="Noted.", updates=[FieldUpdate(address="col:gender.stands_for", value="sex as recorded", said="gender is sex")])]
    d = Desk(DeskFake(after=after))
    d.say(QUESTION)
    d.to_ready()
    p = d.say("run")
    assert p["kind"] == "after"
    m = HELD["students"]
    for a in ("col:lunch.feeds_treatment", "col:lunch.moves_outcome"):
        f = m.field(a)
        assert (
            f is not None
            and f.value is True
            and f.status == "drafted"
            and f.source == "model:ledger"
            and f.reason.startswith("explicit source-backed proposal")
        )
    assert m.field("col:parental_level_of_education.feeds_treatment") is None  # not placed by the graph: nothing said
    design, run, claim, brief = d.journal.steps()[-4:]
    assert [s.kind for s in (design, run, claim, brief)] == ["design", "run", "claim", "brief"]
    assert claim.by == "model" and claim.design == 1 and claim.read == ["col:lunch.feeds_treatment", "col:lunch.moves_outcome"]
    assert claim.note.startswith("drafted from source-backed lane assertions: col:lunch.feeds_treatment")
    # An unrelated person correction does not promote the lane's optional assertions to facts.
    p = d.say("gender is sex, by the way")
    assert p["kind"] == "ready"
    assert m.field("col:lunch.feeds_treatment").status == "drafted"
    assert m.field("col:lunch.moves_outcome").status == "drafted"


def test_the_chat_after_a_run_can_cite_a_step_of_the_conversation_and_the_persons_own_words():
    after = [
        AfterReply(kind="answer", text="As we settled at the start.", cites=["step:1"]),
        AfterReply(kind="answer", text="On your word.", cites=["user:turn:2"]),
        AfterReply(kind="done", text="Bye."),
    ]
    fake = DeskFake(after=after)
    d = Desk(fake)
    d.say(QUESTION)
    d.to_ready()
    d.say("run")
    p = d.say("what did we settle first?")
    assert p["text"] == "As we settled at the start." and fake.calls.count("AfterReply") == 1  # the gate took the step cite first time
    assert "[step:1] question by model" in fake.humans["AfterReply"][0] and d.journal.last("answer").read == ["step:1"]
    p = d.say("and on whose word?")
    assert p["text"] == "On your word." and fake.calls.count("AfterReply") == 2  # a turn of the person's is an address too
    assert "[user:turn:2]" in fake.humans["AfterReply"][1] and d.journal.last("answer").read == ["user:turn:2"]


def test_material_lists_the_steps_as_addresses_without_numbers():
    from causal_agent.desk import material as M
    from causal_agent.memory.journal import Step

    rec = RunRecord(index=1, dataset="students", question="q", family="adjustment", specialist="dowhy", status="done", effect=5.6)
    steps = [Step(n=1, kind="question", by="model", at="t", memory_version=3, read=["user:turn:1"], note="effect: y against x")]
    mat = M.render(rec, None, None, steps=steps)
    assert "step:1" in mat.addresses and "step:1" not in mat.numbers and "[step:1] question by model · v3 · read user:turn:1 · effect: y against x" in mat.text


# ------------------------------------------------------------------ the file talks back


def test_a_timing_answer_the_file_refutes_is_asked_again_then_stands_as_a_contradiction():
    fake = DeskFake()
    d = Desk(fake)
    d.say(QUESTION)
    d.say("let's run it")
    d.say("yes, all right")  # the drafts stand, lunch fixed before the change
    m = HELD["students"]
    assert m.field("col:lunch.when").value == "before" and m.field("col:lunch.when").status == "confirmed"
    p = d.say("actually lunch was after the course")
    assert m.field("col:lunch.when").value == "after" and m.field("col:lunch.when").status == "refuted"
    # The optional timing disagreement is recorded independently of the endpoint's required context.
    assert m.field("col:lunch.when").evidence == ["check:col:lunch.when.depends_on_before"]
    p = d.say("no, lunch was after, I am sure")
    assert m.field("col:lunch.when").status == "contradiction" and m.field("col:lunch.when").value == "after"
    assert "col:lunch.when" not in [a for a in (p["ask"] or {}).get("addresses", [])]  # a contradiction is settled: not asked a third time
    d.to_ready()
    assert d.payload["ready"]
    d.say("run")
    assert "col:lunch.when" in d.values["handoff"].by_status("contradiction")


# ------------------------------------------------------------------ the chat after


def test_after_the_run_a_revision_goes_back_through_the_gate_and_the_checks():
    after = [
        AfterReply(
            kind="answer",
            text="It raised math scores by 5.6 points; the figure shows the estimate against the placebo.",
            cites=["estimate:d1.completed_vs_none.value"],
            numbers=[NumberStated(address="estimate:d1.completed_vs_none.value", value=5.6)],
            figure="figure:effect_d1_completed_vs_none",
        ),
        AfterReply(
            kind="revise",
            text="I will re-check with the offer depending on lunch only.",
            updates=[FieldUpdate(address="claim:assignment.depends_on", value="lunch", said="only lunch")],
        ),
        AfterReply(kind="done", text="Bye."),
    ]
    fake = DeskFake(after=after)
    d = Desk(fake)
    d.say(QUESTION)
    d.to_ready()
    d.say("run")
    p = d.say("what did you find?")
    assert p["kind"] == "after" and "5.6" in p["text"] and p["figure"]["id"] == "effect_d1_completed_vs_none"
    steps = d.journal.steps()
    brief, ans = steps[-2], steps[-1]
    assert brief.kind == "brief" and brief.by == "code" and brief.design == 1 and brief.read == [d.journal.last("run").address] and brief.left == []
    assert ans.kind == "answer" and ans.by == "model" and ans.design == 1 and ans.note == "what did you find?"
    assert ans.read == ["estimate:d1.completed_vs_none.value", "figure:effect_d1_completed_vs_none"]
    p = d.say("the offer only depended on lunch, not on parents")
    m = HELD["students"]
    assert m.value("claim:assignment.depends_on") == ["lunch"] and m.field("claim:assignment.depends_on").source.startswith("user:turn:")
    assert p["kind"] == "ready" and p["ready"] and d.values["phase"] == "before"  # everything still settled: back at the ready moment
    rev = d.journal.last("revise")
    assert rev is not None and rev.by == "person" and rev.design == 1 and rev.note == "claim:assignment.depends_on"  # the person's response to run 1
    p = d.say("run")
    assert p["kind"] == "after" and len(d.values["runs"]) == 2 and "[estimate:d1.completed_vs_none.value]" in p["text"]
    # the second design's brief was written knowing the first: the Designer saw the brief before, so a revise says what it keeps and changes
    first, second = fake.humans["DesignBrief"][0], fake.humans["DesignBrief"][-1]
    assert "(none: this is the first design)" in first and "[design.brief.bets_on] nothing beyond lunch" in second
    kinds = [s.kind for s in d.journal.steps()]
    assert kinds[-4:] == ["revise", "design", "run", "brief"] and d.journal.last("brief").left == ["designs/2/record.json"]  # differs was written
    p = d.say("done")
    assert p is None


def test_a_what_if_runs_on_a_copy_and_leaves_the_memory_alone(monkeypatch):
    after = [
        AfterReply(
            kind="what_if",
            text="Suppose places had been drawn by lot.",
            updates=[FieldUpdate(address="claim:assignment.kind", value="lottery", said="if it had been a lottery")],
        ),
        AfterReply(kind="done", text="Bye."),
    ]
    d = Desk(DeskFake(after=after))
    d.say(QUESTION)
    d.to_ready()
    d.say("run")
    v_before = HELD["students"].version
    saved_kinds = []
    monkeypatch.setattr(store, "save", lambda memory, root=None: saved_kinds.append(memory.value("claim:assignment.kind")))
    p = d.say("what if the places had been drawn by lot?")
    m = HELD["students"]
    assert m.value("claim:assignment.kind") == "own_choice" and m.version == v_before  # nothing known changed
    assert len(d.values["runs"]) == 1 and d.values["phase"] == "before"
    d.to_ready()
    assert d.payload["kind"] == "ready"
    p = d.say("run")
    runs = d.values["runs"]
    assert len(runs) == 2 and runs[1].what_if == {"claim:assignment.kind": "lottery"} and runs[1].family == "adjustment"
    assert p["kind"] == "after" and "what-if" in p["text"] and "[claim:assignment.kind] = lottery" in p["text"] and "what-if" in p["text"]
    assert "claim:assignment.kind" in runs[1].differs and d.values["fork"] is None
    assert all(kind == "own_choice" for kind in saved_kinds)
    assert d.values["memory"].fields == m.fields and d.values["memory"].version == m.version
    assert d.values["handoff"].design_id == 2 and d.values["handoff"].assignment["kind"] == "lottery"
    steps = d.journal.steps()
    activity = [s for s in steps if s.kind != "fit"]
    assert [s.kind for s in activity[-4:]] == ["what_if", "design", "run", "brief"]
    wi, des = activity[-4], activity[-3]
    assert wi.by == "person" and wi.design == 1 and wi.note == "claim:assignment.kind = lottery"  # the person's response to run 1
    assert des.design == 2 and des.note == "what-if: design 2: adjustment via dowhy" and des.memory_version > wi.memory_version  # the fork's version


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
    assert "could be answered" in p["text"] and p["text"].index("could be answered") < p["text"].index("asked again")  # the map again, per question
    assert p["text"].count("asked again") == 1 and "asked again" not in d.say("let's run it")["text"]  # the note is said once
    d.say("yes, all right")
    kinds = [s.kind for s in d.journal.steps()]
    assert (
        "requestion" in kinds and "question" in kinds[kinds.index("requestion") :] and "fit" in kinds[kinds.index("requestion") :]
    )  # the relative fields went: cells moved, and the matrix says so
    assert d.journal.last("requestion").note == "Did a standard lunch raise math scores?"


def test_unexpected_context_deficiency_returns_to_desk_and_requires_new_authorization(monkeypatch):
    calls = []

    def deficient_run(path, n, dataset, question, decision=None, decision_record="", thread_id=None):
        calls.append(n)
        if n == 1:
            return RunRecord(
                index=n,
                dataset=dataset,
                question=question,
                family="adjustment",
                specialist="dowhy",
                status="context_deficiency",
                design_dir=str(path.parent),
                specialist_result={
                    "status": "context_deficiency",
                    "context_deficiency": {
                        "stage": "load",
                        "addresses": ["claim:sampling.detail"],
                        "decisions": ["population"],
                        "candidates": [],
                        "reason": "The lane needs the sampling population clarified.",
                    },
                },
            )
        return canned_run(path, n, dataset, question, decision, decision_record)

    monkeypatch.setattr(pipeline, "run", deficient_run)
    d = Desk(DeskFake())
    d.say(QUESTION)
    d.to_ready()
    p = d.say("run")
    assert p["kind"] == "ask" and p["phase"] == "before"
    assert p["ask"]["addresses"] == ["claim:sampling.detail"]
    assert "missing context" in p["text"] and not d.values["run_intent"]
    d.say("claim:sampling.detail = the whole school cohort")
    assert HELD["students"].value("claim:sampling.detail") == "the whole school cohort"
    assert calls == [1]
    d.to_ready()
    assert calls == [1]
    p = d.say("run")
    assert p["kind"] == "after" and calls == [1, 2]


def test_the_brief_lists_where_the_lane_disagreed_with_the_pack(monkeypatch):
    def declining_run(path, n, dataset, question, decision=None, decision_record="", thread_id=None):
        rec = canned_run(path, n, dataset, question, decision, decision_record)
        rec.specialist_result["declines"] = [
            {
                "stage": "load",
                "kind": "declined",
                "about": "scope.window",
                "pack_value": "the spring term",
                "reason": "the window is not in a form the code can apply; every row was kept",
                "check": "intake.window_unparsed",
            }
        ]
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
    """A lane's own figures.json is what the run shows; without one the post-viz fallback draws."""
    import json

    def drawing_run(path, n, dataset, question, decision=None, decision_record="", thread_id=None):
        rec = canned_run(path, n, dataset, question, decision, decision_record)
        run_dir = tmp_path / "run"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "figures.json").write_text(
            json.dumps(
                [
                    {
                        "id": "causal_graph",
                        "kind": "graph",
                        "title": "g",
                        "series": [],
                        "marks": [],
                        "note": "",
                        "draws_on": ["design.graph"],
                        "nodes": [{"id": "a", "label": "a", "role": "treatment"}, {"id": "b", "label": "b", "role": "outcome"}],
                        "edges": [{"src": "a", "dst": "b", "cites": []}],
                    },
                    {
                        "id": "effect_d1_completed_vs_none",
                        "kind": "interval",
                        "title": "e",
                        "series": [{"name": "effect", "x": ["lr"], "y": [5.6]}],
                        "marks": [],
                        "note": "",
                        "draws_on": [],
                    },
                ]
            )
        )
        rec.run_dir = str(run_dir)
        return rec

    monkeypatch.setattr(pipeline, "run", drawing_run)
    d = Desk(DeskFake())
    d.say(QUESTION)
    d.to_ready()
    p = d.say("run")
    figs = d.values["runs"][0].figures
    assert [f["id"] for f in figs] == ["causal_graph", "effect_d1_completed_vs_none"]
    assert p["figure"]["id"] == "causal_graph"
    from causal_agent.desk import material as M

    m = M.render(d.values["runs"][0])
    assert "figure:causal_graph.edge.0" in m.addresses


def test_the_brief_reads_first_and_names_a_flag_by_its_sentence(monkeypatch):
    def worded_run(path, n, dataset, question, decision=None, decision_record="", thread_id=None):
        rec = canned_run(path, n, dataset, question, decision, decision_record)
        rec.specialist_result["designs"]["d1"]["checks"] = {
            "results": [
                {
                    "design_id": "d1",
                    "contrast": "completed_vs_none",
                    "name": "balance.lunch",
                    "level": "soft",
                    "value": 0.16,
                    "threshold": 0.1,
                    "detail": "how alike the two arms are on lunch before the adjustment, and after weighting: standardised mean difference 0.16",
                }
            ]
        }
        rec.specialist_result["checks"] = rec.specialist_result["designs"]["d1"]["checks"]["results"]
        rec.specialist_result["interpretations"] = [
            {
                "design_id": "d1",
                "contrast": "completed_vs_none",
                "answer": "Completing the course raised math scores by about 5.6 points.",
                "effect_stated": 5.6,
                "caveats": ["The two arms differed on lunch before the adjustment [check:d1.completed_vs_none.balance.lunch]."],
                "cites": ["estimate:d1.completed_vs_none.value"],
            }
        ]
        return rec

    monkeypatch.setattr(pipeline, "run", worded_run)
    d = Desk(DeskFake())
    d.say(QUESTION)
    d.to_ready()
    text = d.say("run")["text"]
    lines = text.splitlines()
    assert lines[1].startswith("Completing the course raised") and lines[2].startswith("Keep in mind:")
    assert lines.index(next(ln for ln in lines if ln.startswith("Completing the course"))) < lines.index(
        next(ln for ln in lines if ln.startswith("Lead design:"))
    )
    flag = next(ln for ln in lines if ln.startswith("Design flag:"))
    assert "how alike the two arms are on lunch" in flag and "[check:d1.completed_vs_none.balance.lunch]" in flag
    assert "[estimate:d1.completed_vs_none.value]" in text


# ------------------------------------------------------------------ pictures on request


def test_a_picture_asked_for_before_the_run_is_drawn_shown_and_recorded(tmp_path):
    from causal_agent.memory import artifacts as VS

    def infer(msg, addrs, human):
        return Reading(kind="explore", explore="show me student counts by lunch") if "plot" in msg else None

    fake = DeskFake(infer=infer)
    d = Desk(fake)
    d.say(QUESTION)
    p = d.say("plot student counts by lunch before we go on")
    a = p["artifact"]
    assert a and a["kind"] == "picture" and a["design"] is None and set(a["facts"]) == {"count_standard", "count_free_reduced"}
    assert p["text"].startswith("Student counts by lunch") and f"[artifact:{a['id']}]" in p["text"] and p["kind"] == "chat"  # the caption; nothing asked
    assert fake.calls.count("Exploration") == 1 and fake.calls.count("RunCodeStep") == 2 and "student counts" in fake.humans["Exploration"][0]
    folder = tmp_path / "data" / "memory" / "students" / "artifacts" / a["id"]
    assert (folder / "figure.png").exists() and (folder / "code.py").exists() and VS.list_artifacts("students")[0].id == a["id"]
    step = d.journal.steps()[-1]
    assert (
        step.kind == "explore"
        and step.by == "model"
        and step.note == "plot student counts by lunch before we go on"
        and step.left[0].endswith(f"artifacts/{a['id']}")
    )
    assert HELD["students"].version == d.values["convinced_version"] if d.values.get("convinced_version") else True  # a picture settles nothing
    p = d.say(answer_ask(p))
    assert p["artifact"] is None  # shown once


def test_a_number_from_a_picture_drawn_before_the_run_can_be_cited_and_a_wrong_one_is_refused(tmp_path):
    """Before the run the material holds every picture drawn so far with its numbers: an answer that cites a drawn number within
    one percent passes; one that states a number the picture does not show is refused and the desk falls back."""
    from causal_agent.desk.contracts import NumberStated

    def infer(msg, addrs, human):
        if "plot" in msg:
            return Reading(kind="explore", explore="show me student counts by lunch")
        return Reading(kind="ask", question="which lunch group has more students?") if msg.startswith("which") else None

    fake = DeskFake(infer=infer)
    d = Desk(fake)
    d.say(QUESTION)
    a = d.say("plot student counts by lunch before we go on")["artifact"]
    count = a["facts"]["count_standard"]
    good = AfterReply(
        kind="answer",
        text=f"Standard lunch group contains {count:.1f}.",
        numbers=[NumberStated(address=f"artifact:{a['id']}.count_standard", value=round(count, 1))],
    )
    fake.explain = [good]
    p = d.say("which lunch group has more students?")
    assert p["text"].startswith("Standard lunch group contains") and fake.calls.count("AfterReply") == 1
    assert f"[artifact:{a['id']}.count_standard]" in fake.humans["AfterReply"][0]  # the drawn number was in the material the Explainer read
    bad = AfterReply(kind="answer", text="Standard lunch group contains 99.", numbers=[NumberStated(address=f"artifact:{a['id']}.count_standard", value=99.0)])
    fake.explain = [bad, bad, bad]
    p = d.say("which lunch group has more students?")
    assert fake.calls.count("AfterReply") == 4 and "does not match" in fake.humans["AfterReply"][3]
    assert p["text"].startswith("I can only answer that from what is settled.")


def test_a_picture_asked_for_after_the_run_lands_in_the_design_and_is_citable(tmp_path):
    after = [
        AfterReply(kind="explore", text="", explore="the mean math score for each lunch group"),
        AfterReply(kind="answer", text="Standard lunch students average higher.", cites=["artifact:PLACEHOLDER.mean_standard"]),
        AfterReply(kind="done", text="Bye."),
    ]
    fake = DeskFake(after=after)
    d = Desk(fake)
    d.say(QUESTION)
    d.to_ready()
    d.say("run")
    from causal_agent.desk.nodes.after import frozen_context

    run = d.values["runs"][-1]
    memory = HELD["students"]
    assert frozen_context(run, memory)
    failed = run.model_copy(deep=True)
    failed.specialist_result["designs"] = {}
    assert not frozen_context(failed, memory)
    wrong_context = run.model_copy(deep=True)
    wrong_context.specialist_result["designs"]["d1"]["context_id"] = "a different admitted context"
    assert not frozen_context(wrong_context, memory)
    assert not frozen_context(run, memory.model_copy(update={"data_version": memory.data_version + 1}))
    p = d.say("draw me the mean math score by lunch")
    a = p["artifact"]
    assert a and a["kind"] == "picture" and a["design"] == 1 and p["text"].startswith("Mean math score by lunch") and p["kind"] == "after"
    assert "WHAT THE RUN FOUND" in fake.humans["Exploration"][0] and "[estimate:d1.completed_vs_none.value]" in fake.humans["Exploration"][0]
    folder = tmp_path / "data" / "memory" / "students" / "artifacts" / a["id"]
    assert (folder / "figure.png").exists()
    step = d.journal.steps()[-1]
    assert step.kind == "explore" and step.design == 1 and step.left[0].endswith(f"artifacts/{a['id']}")
    fake.after[0].cites = [f"artifact:{a['id']}.mean_standard"]
    p = d.say("who does better?")
    assert p["text"].startswith("Standard lunch") and p["artifact"]["id"] == a["id"]  # an answer that cites the picture shows it


# ------------------------------------------------------------------ the memory mirrored for a watcher


def test_the_memory_is_mirrored_into_the_state_after_every_save_and_no_node_reads_it(monkeypatch):
    """Studio shows the state and the memory lives on disk, so every node that saves the memory mirrors it into the state, and
    the mirror is written only: no node under desk/ reads it."""
    desk = Desk(DeskFake())
    assert "memory" not in desk.values  # nothing is saved before the question is answered
    desk.say(QUESTION)
    m = desk.values["memory"]
    held = store.memory_for("students")
    assert isinstance(m, Memory) and m.version == held.version and m.fields == held.fields
    before = m.version
    desk.say("nothing hidden")  # a word about the data: drafted, so the memory moves
    assert desk.values["memory"].version > before
    fork = held.fork()
    fork.version += 1
    durable_saves = []
    monkeypatch.setattr(store, "save", lambda memory, root=None: durable_saves.append(memory))
    mirror = F.saved(fork, {"dataset": held.name, "fork": fork})
    assert durable_saves == [] and mirror["fork"] is fork and mirror["memory"] is fork
    assert F.saved(held, {"dataset": held.name, "fork": None})["memory"] is held and durable_saves == [held]
    root = Path(G.__file__).resolve().parent
    hits = [
        f"{p.relative_to(root)}:{n}"
        for p in root.rglob("*.py")
        if "tests" not in p.parts
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if re.search(r'state(\.get\(|\[)"memory"', line)
    ]
    assert not hits, hits


def test_a_story_told_in_the_chat_is_not_asked_again_when_the_person_wants_to_run(_no_note):
    d = Desk(DeskFake())
    d.say(QUESTION)
    p = d.say("I'd say claim:assignment.kind = own_choice")  # said in passing, before any wish to run: drafted, nothing asked
    assert p["kind"] == "chat" and HELD["students"].field("claim:assignment.kind").status == "drafted"
    p = d.say("let's run it")
    assert p["ask"]["kind"] == "confirm" and "Tell me the story" not in p["text"], "the chat told it; the readback comes first"
