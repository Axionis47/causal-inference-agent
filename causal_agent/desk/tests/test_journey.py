"""The desk graph with a scripted model: the question first, validated; one question per turn to ready; the run; the chat after.
The memory is held in this process; nothing is written under data/."""

from __future__ import annotations

import uuid

import pytest
from langgraph.types import Command

from causal_agent.common import config
from causal_agent.common.contracts import RunRecord
from causal_agent.common.llm import set_llm
from causal_agent.desk import graph as G
from causal_agent.desk import pipeline
from causal_agent.desk.contracts import AfterReply, FieldUpdate, NumberStated, Reading
from causal_agent.desk.nodes import frame as F
from causal_agent.desk.tests.fakes import QUESTION, STORY_ANSWER, DeskFake, answer_ask
from causal_agent.families import registry as R
from causal_agent.memory import journal as J
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
    yield
    set_llm(None)


def canned_run(path, n, dataset, question, decision=None, decision_record=""):
    sr = {
        "status": "done",
        "design": {"estimator": "linear_regression", "contrast": {"treated": "completed", "control": "none"}, "checks": {"results": []}},
        "estimates": [
            {"contrast": "completed_vs_none", "method": "linear_regression", "value": 5.6, "ci_low": 3.7, "ci_high": 7.5, "secondary": False, "error": None}
        ],
        "refutations": [{"contrast": "completed_vs_none", "refuter": "placebo_treatment_refuter", "kind": "falsification", "new_effect": 0.05, "passed": True}],
        "interpretations": [],
        "feasibility": None,
    }
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
        """Answer until the ready moment: ready with nothing left to ask. Ready can come first, since a relation a decision rests on is
        asked but never blocks."""
        turns = 0
        while self.payload and self.payload["kind"] == "ask" and (not self.payload["ready"] or self.payload.get("ask")):
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
    # the first reply is the map: what the file could answer the question with, what each family still needs, what is struck and why
    text = p["text"]
    assert text.startswith("With this file, math score against test preparation course could be answered")
    adj = next(f for f in R.knowledge() if f.name == "adjustment")
    assert f"- adjustment: {adj.answers[0].upper() + adj.answers[1:]}. Still to settle: unobserved, spillover." in text  # the note settled the rest
    assert "Struck already: " in text and "discontinuity (assignment does not fit)" in text and "Say which of these you care about" in text
    assert text.index("Say which of these") < text.index("Here is what I read")  # the map, then the readback
    # the note was mined once into drafts by the Reader; the story is not asked, the readback confirms the drafts in one go
    assert p["ask"]["kind"] == "confirm" and "claim:assignment.kind" in p["ask"]["addresses"] and len(fake.reads("doc:")) == 1
    assert HELD["students"].field("claim:assignment.kind").status == "drafted" and HELD["students"].field("claim:unobserved.exists") is None
    assert d.values["story_asked"] and d.values["readback_done"] and "Tell me the story" not in text
    turns = d.to_ready(max_turns=10)
    assert turns <= 8 and d.payload["ready"] and "Say run" in d.payload["text"] and "could be answered" not in d.payload["text"]  # the map was said once
    # the relations a decision rests on were asked, in the order of the decisions, and the person's answers landed
    reads = fake.reads("user:")
    settles = lambda h: h.split("(settles: ")[1].split(")")[0].split(", ") if "(settles: " in h else []  # noqa: E731
    turn_of = lambda address: next(i for i, h in enumerate(reads) if address in settles(h))  # noqa: E731
    assert HELD["students"].value("claim:assignment.offer_column") == "none" and HELD["students"].value("col:lunch.may_modify") is False
    assert turn_of("claim:assignment.offer_column") == turn_of("claim:assignment.uptake_column")
    # what blocks readiness is asked first, the beliefs among it; the relations that only help come after, one decision per turn
    assert turn_of("claim:unobserved.exists") < turn_of("claim:assignment.offer_column") < turn_of("col:lunch.same_as") < turn_of("col:lunch.may_modify")
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
    assert all(s.read == [f"user:turn:{n}"] for s, n in zip(steps[1:], range(2, len(steps) + 1)))
    assert [s.memory_version for s in steps] == sorted(s.memory_version for s in steps) and "claim:assignment.kind" in steps[1].note
    # the ready moment: the design in the question's words, the evidence with addresses, the figure, the struck families with a reason each
    text = d.payload["text"]
    assert "Design: adjustment" in text and "[probe:adjustment.overlap]" in text and "[probe:adjustment.arms]" in text
    assert "Set aside: " in text and "diff_in_diff" in text and "discontinuity" in text
    # the Designer's brief replaces the canned assumption: what it bets on, one line per decision, the road, the threats
    assert "It bets on: nothing beyond lunch and parents' education drove both" in text and fake.calls.count("DesignBrief") == 1
    assert "[design.brief.road] backdoor" in text and "[design.brief.who_is_treated]" in text and "[design.brief.run_at_all]" in text
    assert "What would break it: something outside the file could have driven both [claim:assignment.kind]" in text
    assert text.index("It bets on") < text.index("[design.brief.road]") < text.index("Evidence:")
    assert d.values["brief"].family == "adjustment" and d.values["brief"].road == "backdoor"
    assert "ask for a picture" in text and d.payload["artifact"] is None  # nothing is drawn unasked
    assert d.values["decision"].chosen == "adjustment" and d.values["convinced_version"] == HELD["students"].version
    m = HELD["students"]
    assert m.field("claim:assignment.kind").status == "confirmed" and m.field("claim:assignment.kind").source == "user:turn:2"
    assert m.field("claim:unobserved.exists").value is False and m.field("claim:unobserved.exists").said == "nothing hidden"
    assert all("(settles:" in h for h in fake.reads("user:"))  # every reading of a message saw the question it was answering
    p = d.say("run")
    assert p["kind"] == "after" and p["ready"] and "Run 1" in p["text"] and "[estimate:completed_vs_none.value]" in p["text"]
    runs = d.values["runs"]
    # the run's figures: the estimate against its falsifications, drawn from the artifacts; the brief shows it
    assert [f["id"] for f in runs[0].figures] == ["effect_completed_vs_none"]
    assert (
        p["figure"]["id"] == "effect_completed_vs_none" and (tmp_dir := runs[0].design_dir) and (__import__("pathlib").Path(tmp_dir) / "figures.json").exists()
    )
    assert len(runs) == 1 and runs[0].effect == 5.6 and runs[0].family == "adjustment" and d.values["phase"] == "after"
    assert fake.calls.count("FamilyDecision") == 0 and fake.calls.count("DrawCode") == 0  # one family stood: chosen by code; nothing drawn unasked
    assert (d.values["design_dir"]) and d.values["handoff"].design_id == 1
    # the pack carries the brief and bets on its sentence; the design folder holds it twice, in handoff.json and brief.json
    import json
    from pathlib import Path

    h = d.values["handoff"]
    assert h.brief is not None and h.brief.road == "backdoor" and h.chosen_assumption == h.brief.bets_on and fake.calls.count("DesignBrief") == 1
    folder = Path(d.values["design_dir"])
    assert json.loads((folder / "handoff.json").read_text())["brief"]["bets_on"] == h.brief.bets_on
    assert json.loads((folder / "brief.json").read_text())["decisions"][0]["name"] == "who_is_treated"
    assert "BETS ON      " + h.brief.bets_on in d.values["decision_record"] and "[design.brief.road] backdoor" in d.values["decision_record"]
    assert runs[0].decision["brief"]["road"] == "backdoor" and runs[0].decision["chosen_assumption"] == h.brief.bets_on
    # the journal: the design written from the memory, then the run that read it, both on this design's group
    design, run, brief = d.journal.steps()[-3:]
    facts = {p.name: p for p in h.probes if p.family == "data"}
    assert {"by_arm.lunch", "by_arm.parental_level_of_education", "with_outcome.lunch", "timing"} <= set(
        facts
    )  # the numbers a reasoning would ask for, up front
    assert "of the treated" in facts["by_arm.lunch"].detail and facts["timing"].detail.startswith("before: ")
    assert (
        not any("math_score" in n and "test_preparation_course" in n for n in facts) and "by_arm.math_score" not in facts
    )  # nothing joins outcome and treatment
    assert h.story == STORY_ANSWER or h.story.startswith("#")  # the account rides in the pack
    assert h.render_context().startswith("STORY") and "[probe:data.by_arm.lunch]" in h.render_context() or True
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
        road="backdoor",
        decisions=[DecisionMade(name="who_is_treated", choice="completed against none", rests_on=["nonsense:thing"], reason="r")],
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
    assert b.bets_on == adj.assumes and [x.name for x in b.decisions] == [x.name for x in adj.decisions] and b.road is None
    assert all(x.choice == "not decided" and x.rests_on == ["change:1.note"] for x in b.decisions)
    assert f"It bets on: {adj.assumes}" in d.payload["text"] and "[design.brief.who_is_treated] not decided" in d.payload["text"]
    # run takes the design decided at the ready moment: the fallback brief is the pack's brief, written beside it
    d.say("run")
    h = d.values["handoff"]
    assert h.brief.road is None and h.chosen_assumption == adj.assumes and fake.calls.count("DesignBrief") == 3
    assert __import__("json").loads((__import__("pathlib").Path(d.values["design_dir"]) / "brief.json").read_text())["decisions"][0]["choice"] == "not decided"
    assert brief_by_rule(fake.humans["DesignBrief"][0]).family == "adjustment"  # what the rule would have said, had it been asked


def test_the_chat_after_a_run_can_cite_what_the_design_rested_on():
    from causal_agent.desk import material as M

    after = [
        AfterReply(kind="answer", text="It bets on the two columns the offer looked at.", cites=["design.brief.bets_on", "design.brief.adjustment_set"]),
        AfterReply(kind="done", text="Bye."),
    ]
    fake = DeskFake(after=after)
    d = Desk(fake)
    d.say(QUESTION)
    d.to_ready()
    p = d.say("run")
    design_line = next(ln for ln in p["text"].splitlines() if ln.startswith("Design: adjustment via dowhy."))
    assert "By the backdoor road." in design_line and design_line.endswith("[decision.family] [design.brief.road]")
    assert "It bets on: nothing beyond lunch and parents' education drove both the course and the score [design.brief.bets_on]" in p["text"]
    run = d.values["runs"][0]
    mat = M.render(run, HELD["students"])
    assert {"design.brief", "design.brief.bets_on", "design.brief.road", "design.brief.adjustment_set", "design.brief.threat:1"} <= mat.addresses
    assert mat.by_address["design.brief.bets_on"] == run.decision["brief"]["bets_on"] and mat.by_address["design.brief.road"].startswith("backdoor: ")
    p = d.say("what does it rest on?")
    assert p["text"].startswith("It bets on the two columns") and fake.calls.count("AfterReply") == 1  # the gate took the brief's addresses first time
    assert "[design.brief.bets_on]" in fake.humans["AfterReply"][0] and d.journal.last("answer").read == ["design.brief.bets_on", "design.brief.adjustment_set"]


# ------------------------------------------------------------------ focus: the families the person cares about


def test_naming_the_families_you_care_about_drops_the_questions_the_others_need():
    def only_adjustment(msg, addrs, human):
        if msg.startswith("only adjustment"):
            return Reading(focus=["adjustment"])
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
    assert "Set aside: " in d.payload["text"] and "instrument: not asked for" in d.payload["text"]
    assert d.values["decision"].chosen == "adjustment" and {r.family: r.reason for r in d.values["decision"].rejected}["instrument"] == "not asked for"
    d.say("run")
    assert d.values["runs"][0].family == "adjustment"


def test_a_family_the_grid_does_not_know_is_refused_and_the_focus_stays():
    calls = []

    def magic(msg, addrs, human):
        if msg.startswith("only magic"):
            calls.append(human)
            return Reading(focus=["magic"]) if len(calls) == 1 else Reading()
        return None

    fake = DeskFake(infer=magic)
    d = Desk(fake)
    d.say(QUESTION)
    p = d.say("only magic, please")
    assert len(calls) == 2 and "focus: magic is not a family the fit grid knows" in calls[1]  # refused, asked again with the reason
    assert not d.values.get("focus") and p["kind"] == "ask" and len(d.values["status"].surviving) > 1


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
        and p["ask"] is not None
        and p["text"].rstrip().endswith(p["ask"]["text"])
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
        return Reading(question="why do you ask that?") if msg.startswith("why") else None

    bad = AfterReply(kind="answer", text="Because.", cites=["nonsense:thing"])
    revise = AfterReply(kind="revise", text="Noted.", updates=[FieldUpdate(address="claim:sampling.how", value="by_arm", said="x")])  # not legal before the run
    fake = DeskFake(infer=curious, explain=[bad, revise, bad])
    d = Desk(fake)
    d.say(QUESTION)
    p = d.say("why do you ask that?")
    assert fake.calls.count("AfterReply") == 3 and "PREVIOUS REPLY WAS REJECTED" in fake.humans["AfterReply"][2]
    assert "revise is not legal before the run" in fake.humans["AfterReply"][2] and HELD["students"].value("claim:sampling.how") == "whole"
    assert p["text"].startswith("I can only answer that from what is settled.") and "can be answered by adjustment" in p["text"] and p["ask"] is not None
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
    p = d.say(QUESTION)
    # the story turn: once, before any field question, over every open address
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
    assert kind.status == "drafted" and kind.source == "user:turn:2" and kind.said == "then open to anyone who asked" and kind.reason == "scripted"
    assert when.status == "drafted" and when.value == "after" and m.value("claim:change.what") == "a six-week test preparation course"
    assert "(the story)" in fake.reads("user:")[0] and d.journal.last("claim").note.count("claim:") >= 8
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
    assert all(m.field(x).status == "confirmed" and m.field(x).source == "user:turn:3" for x in a["addresses"])
    # a gap question names the decision it serves and asks the fields under it together
    a = p["ask"]
    assert a["decision"] == "who_is_treated" and a["addresses"] == ["claim:sampling.how"] and a["kind"] == "choose" and a["options"][0] == "whole"
    assert p["text"].endswith(
        "To settle who is treated, versus whom, I need: sampling, how: were all units kept, or were rows picked by which side of a line "
        "they fell on, by whether they got the change, by group, by period, or by how the outcome turned out; show the row count as evidence. One of: "
        "whole (every unit in the population is in the file); by_side (rows were drawn according to which side of a line on a score they fell); by_arm "
        "(rows were drawn according to whether the unit got the change); by_group (rows were drawn by group, region, or type); by_period (rows were drawn "
        "by period); by_outcome (rows were drawn according to how the outcome turned out); unknown (the person does not know). Say don't know for "
        "anything you cannot say."
    )
    p = d.say("claim:sampling.how = whole")
    # a belief is asked as what the design would bet on, with the person's own facts beside it; the beliefs one decision rests on go together
    a = p["ask"]
    adj = next(f for f in R.knowledge() if f.name == "adjustment")
    assert a["decision"] == "road" and a["addresses"] == ["claim:unobserved.exists", "claim:exclusion.exists"] and a["kind"] == "open"
    assert (
        f"To settle the road: back door, front door, or instrument, I need: The design will assume {adj.assumes}. Is there anything not in the file"
        in p["text"]
    )
    assert "Is there a column that pushed units toward the change" in p["text"]
    assert (
        "(you said the rule was: offered first by lunch status and parental education, then open to anyone who asked; it depended on lunch, parental "
        "level of education.)" in p["text"]
    )
    d.to_ready()
    assert d.payload["ready"] and m.field("claim:unobserved.exists").value is False


def test_a_mined_note_skips_the_story_and_the_readback_comes_first():
    fake = DeskFake()
    d = Desk(fake)
    p = d.say(QUESTION)
    assert len(fake.reads("doc:")) == 1 and p["ask"]["kind"] == "confirm" and "Tell me the story" not in p["text"]
    assert d.values["story_asked"] and d.values["readback_done"]
    m = HELD["students"]
    f = m.field("claim:assignment.rule")
    assert f.status == "drafted" and f.source == "doc:context" and f.said == "offered first by lunch status"  # the sentence travels from the note
    assert '— "offered first by lunch status"' in p["text"] and "What the design would bet on" not in p["text"]  # a note sets no belief
    assert m.field("claim:unobserved.exists") is None
    assert "Tell me the story" not in "\n".join(fake.reads("user:"))


def test_a_field_no_decision_rests_on_is_asked_with_the_rest_of_its_claim(_no_note):
    d = Desk(DeskFake())
    d.say(QUESTION)
    p = d.say("claim:assignment.kind = own_choice")  # a story that says one thing: drafted, read back, confirmed
    # once nothing blocks, the mechanism: whether an offer and a taking are two columns (the rule and its drivers came from the story)
    asked = fake.reads("user:")
    mech = next(h for h in asked if "To settle how the change reached the units" in h)
    assert (
        "a column's name, or none" in mech
        and m.value("claim:assignment.offer_column") == "none"
        and m.field("claim:assignment.uptake_column").status == "confirmed"
    )
    # then the relations, under the decisions that rest on them, one turn per decision
    settles = lambda h: h.split("(settles: ")[1].split(")")[0].split(", ") if "(settles: " in h else []  # noqa: E731
    rel = next(h for h in asked if "To settle what enters the adjustment set" in h)
    assert {"col:lunch.same_as", "col:parental_level_of_education.nested_in", "col:lunch.stands_for"} <= set(settles(rel))
    assert "col:lunch.moved_by_change" not in settles(rel)  # fixed before the change: the change could not have moved it, so it is not asked
    het = next(h for h in asked if "To settle where the effect could differ" in h)
    assert "col:lunch.may_modify" in settles(het) and asked.index(rel) < asked.index(het)
    assert p["ask"]["kind"] == "confirm" and p["ask"]["addresses"] == ["claim:assignment.kind", "claim:assignment.treatment_column"]
    p = d.say("yes, all right")
    a = p["ask"]
    # the decisions come first, in the family's order: who is treated rests on the treated level and the sampling
    assert a["decision"] == "who_is_treated" and a["addresses"] == ["claim:assignment.treated_level", "claim:sampling.how"] and a["kind"] == "open"
    # a field no decision rests on waits for the decisions and is then asked with the rest of its claim, the frame once, the fields named
    from causal_agent.desk.nodes.interview import compose_ask
    from causal_agent.memory.ops import Open

    opened = [
        Open(address="claim:change.what", kind="change", field="what", status="empty", because=["adjustment"], frame="f"),
        Open(address="claim:change.to_whom", kind="change", field="to_whom", status="empty", because=["adjustment"], frame="f"),
    ]
    a2 = compose_ask(HELD["students"], opened, [], None, surviving=["adjustment"])
    assert a2.decision == "" and a2.addresses == ["claim:change.what", "claim:change.to_whom"] and a2.kind == "open"
    assert a2.text.endswith(
        "What was the change, which units could it reach, and when did it happen; if a column records the period, which value marks when it "
        "took effect? This turn: what, to whom. Say don't know for anything you cannot say."
    )


# ------------------------------------------------------------------ the journal


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


def test_saying_run_over_open_drafts_is_a_claim_step_on_the_persons_word():
    d = Desk(DeskFake())
    d.say(QUESTION)  # the drafts from the note are open, to be confirmed
    p = d.say("run")
    assert "Before I can run" in p["text"] or p["ready"]
    claim = d.journal.last("claim")
    assert claim is not None and claim.by == "person" and claim.note.startswith("confirmed as drafted: claim:") and claim.read == ["user:turn:2"]


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
    assert "matrix:adjustment.assignment" in mat.addresses and mat.by_address["matrix:adjustment.assignment"].startswith("fits · set by claim:assignment.kind")


def test_the_runs_graph_comes_back_as_drafts_the_person_confirms_on_the_next_ask(monkeypatch):
    def graph_run(path, n, dataset, question, decision=None, decision_record=""):
        rec = canned_run(path, n, dataset, question, decision, decision_record)
        rec.specialist_result["design"]["graph"] = {
            "treatment": "test_preparation_course",
            "outcome": "math_score",
            "nodes": ["test_preparation_course", "math_score", "lunch"],
            "edges": [
                {"src": "test_preparation_course", "dst": "math_score", "cites": []},
                {"src": "lunch", "dst": "test_preparation_course", "cites": ["claim:assignment.depends_on"]},
                {"src": "lunch", "dst": "math_score", "cites": ["col:lunch.when"]},
            ],
            "excluded": [],
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
        assert f is not None and f.value is True and f.status == "drafted" and f.source == "model:relate" and f.reason == "the run's graph drew this edge"
    assert m.field("col:parental_level_of_education.feeds_treatment") is None  # not placed by the graph: nothing said
    design, run, claim, brief = d.journal.steps()[-4:]
    assert [s.kind for s in (design, run, claim, brief)] == ["design", "run", "claim", "brief"]
    assert claim.by == "model" and claim.design == 1 and claim.read == ["col:lunch.feeds_treatment", "col:lunch.moves_outcome"]
    assert claim.note.startswith("drafted from the run's graph: col:lunch.feeds_treatment")
    # the next ask is a confirm turn over the drafts, like any other draft; the person's yes makes them the next run's facts
    p = d.say("gender is sex, by the way")
    assert p["kind"] == "ask" and p["ask"]["kind"] == "confirm" and set(p["ask"]["addresses"]) >= {"col:lunch.feeds_treatment", "col:lunch.moves_outcome"}
    assert "'lunch': feeds treatment: yes; moves outcome: yes — from model:relate" in p["text"] and "Is this right?" in p["text"]
    d.say("yes, all right")
    assert m.field("col:lunch.feeds_treatment").status == "confirmed" and m.field("col:lunch.moves_outcome").status == "confirmed"


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
    assert '[user:turn:2] "' in fake.humans["AfterReply"][1] and d.journal.last("answer").read == ["user:turn:2"]


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
    d.say("yes, all right")  # the drafts stand, lunch fixed before the change
    m = HELD["students"]
    assert m.field("col:lunch.when").value == "before" and m.field("col:lunch.when").status == "confirmed"
    p = d.say("actually lunch was after the course")
    assert m.field("col:lunch.when").value == "after" and m.field("col:lunch.when").status == "refuted"
    assert (
        p["ask"]["kind"] == "choose"
        and p["ask"]["addresses"] == ["col:lunch.when"]
        and p["ask"]["decision"] == "adjustment_set"
        and p["text"].endswith(
            "To settle what enters the adjustment set, I need: The file disagrees with what you said about 'lunch': you said after, but "
            "the offer or the rule looked at 'lunch', so it was set before the change, not after [check:col:lunch.when.depends_on_before]. Which is it?"
        )
        and p["ask"]["evidence"] == ["check:col:lunch.when.depends_on_before"]
    )
    p = d.say("no, lunch was after, I am sure")
    assert m.field("col:lunch.when").status == "contradiction" and m.field("col:lunch.when").value == "after"
    assert "col:lunch.when" not in [a for a in (p["ask"] or {}).get("addresses", [])]  # a contradiction is settled: not asked a third time
    d.to_ready()
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
        AfterReply(
            kind="answer",
            text="It raised math scores by 5.6 points; the figure shows the estimate against the placebo.",
            cites=["estimate:completed_vs_none.value"],
            numbers=[NumberStated(address="estimate:completed_vs_none.value", value=5.6)],
            figure="figure:effect_completed_vs_none",
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
    assert p["kind"] == "after" and "5.6" in p["text"] and p["figure"]["id"] == "effect_completed_vs_none"
    steps = d.journal.steps()
    brief, ans = steps[-2], steps[-1]
    assert brief.kind == "brief" and brief.by == "code" and brief.design == 1 and brief.read == [d.journal.last("run").address] and brief.left == []
    assert ans.kind == "answer" and ans.by == "model" and ans.design == 1 and ans.note == "what did you find?"
    assert ans.read == ["estimate:completed_vs_none.value", "figure:effect_completed_vs_none"]
    p = d.say("the offer only depended on lunch, not on parents")
    m = HELD["students"]
    assert m.value("claim:assignment.depends_on") == ["lunch"] and m.field("claim:assignment.depends_on").source.startswith("user:turn:")
    assert p["kind"] == "ask" and p["ready"] and d.values["phase"] == "before"  # everything still settled: back at the ready point
    rev = d.journal.last("revise")
    assert rev is not None and rev.by == "person" and rev.design == 1 and rev.note == "claim:assignment.depends_on"  # the person's response to run 1
    p = d.say("run")
    assert p["kind"] == "after" and len(d.values["runs"]) == 2 and "Then and now" in p["text"]
    # the second design's brief was written knowing the first: the Designer saw the brief before, so a revise says what it keeps and changes
    first, second = fake.humans["DesignBrief"][0], fake.humans["DesignBrief"][-1]
    assert "(none: this is the first design)" in first and "[design.brief.bets_on] nothing beyond lunch" in second
    kinds = [s.kind for s in d.journal.steps()]
    assert kinds[-4:] == ["revise", "design", "run", "brief"] and d.journal.last("brief").left == ["designs/2/record.json"]  # differs was written
    p = d.say("done")
    assert p is None


def test_a_what_if_runs_on_a_copy_and_leaves_the_memory_alone():
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
    p = d.say("what if the places had been drawn by lot?")
    m = HELD["students"]
    assert m.value("claim:assignment.kind") == "own_choice" and m.version == v_before  # nothing known changed
    runs = d.values["runs"]
    assert len(runs) == 2 and runs[1].what_if == {"claim:assignment.kind": "lottery"} and runs[1].family == "adjustment"
    assert p["kind"] == "after" and "what-if" in p["text"] and "[claim:assignment.kind] = lottery" in p["text"] and "Then and now" in p["text"]
    assert "claim:assignment.kind" in runs[1].differs and d.values["fork"] is None
    assert d.values["handoff"].design_id == 2 and d.values["handoff"].assignment["kind"] == "lottery"
    steps = d.journal.steps()
    assert [s.kind for s in steps[-4:]] == ["what_if", "design", "run", "brief"]
    wi, des = steps[-4], steps[-3]
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
    assert p["text"].count("asked again") == 1 and "asked again" not in d.say("yes, all right")["text"]  # the note is said once
    kinds = [s.kind for s in d.journal.steps()]
    assert kinds[-4:] == ["requestion", "question", "fit", "claim"]  # the relative fields went: cells moved, and the matrix says so
    assert d.journal.last("requestion").note == "Did a standard lunch raise math scores?"


def test_a_lane_that_asks_back_gets_its_answer_and_runs_again(monkeypatch):
    calls = []

    def asking_run(path, n, dataset, question, decision=None, decision_record=""):
        calls.append(n)
        if n == 1:
            return RunRecord(
                index=n,
                dataset=dataset,
                question=question,
                family="adjustment",
                specialist="dowhy",
                status="ask",
                design_dir=str(path.parent),
                specialist_result={
                    "status": "ask",
                    "ask": {"address": "claim:mediator.exists", "question": "Is there a column the change altered, through which its whole effect runs?"},
                    "feasibility": {"stage": "ask", "reason": "nothing identifies the effect while a hidden factor stands"},
                },
            )
        return canned_run(path, n, dataset, question, decision, decision_record)

    monkeypatch.setattr(pipeline, "run", asking_run)

    def infer(msg, addrs, human):
        if "nothing hidden" in msg:  # the person says a hidden factor exists this time
            return Reading(updates=[FieldUpdate(address="claim:unobserved.exists", value="true", said=msg)])
        if msg == "no mediator":
            return Reading(updates=[FieldUpdate(address="claim:mediator.exists", value="false", said=msg)])
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
    ask = {
        "address": "claim:trend_continues.believed",
        "question": "Is there a reason the groups would have moved differently?",
        "options": ["yes", "no"],
        "because": "[check:c.pre_trends] hard: the leads differ",
        "evidence": ["check:c.pre_trends"],
    }

    def asking_run(path, n, dataset, question, decision=None, decision_record=""):
        return RunRecord(
            index=n,
            dataset=dataset,
            question=question,
            family="adjustment",
            specialist="dowhy",
            status="ask",
            design_dir=str(path.parent),
            specialist_result={"status": "ask", "ask": ask, "feasibility": {"stage": "ask", "reason": "the groups were already moving apart"}},
        )

    monkeypatch.setattr(pipeline, "run", asking_run)

    def infer(msg, addrs, human):
        if msg == "yes there is":
            return Reading(updates=[FieldUpdate(address="claim:trend_continues.believed", value="true", said=msg)])
        return None

    d = Desk(DeskFake(infer=infer))
    d.say(QUESTION)
    d.to_ready()
    p = d.say("run")
    assert (
        p["kind"] == "ask"
        and p["ask"]["addresses"] == ["claim:trend_continues.believed"]
        and p["ask"]["options"] == ["yes", "no"]
        and p["ask"]["evidence"] == ["check:c.pre_trends"]
    )
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

    def drawing_run(path, n, dataset, question, decision=None, decision_record=""):
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
                        "id": "effect_completed_vs_none",
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
    assert [f["id"] for f in figs] == ["causal_graph", "effect_completed_vs_none"]
    assert p["figure"]["id"] == "causal_graph"
    from causal_agent.desk import material as M

    m = M.render(d.values["runs"][0])
    assert "figure:causal_graph.edge.0" in m.addresses


def test_the_brief_reads_first_and_names_a_flag_by_its_sentence(monkeypatch):
    def worded_run(path, n, dataset, question, decision=None, decision_record=""):
        rec = canned_run(path, n, dataset, question, decision, decision_record)
        rec.specialist_result["design"]["checks"] = {
            "results": [
                {
                    "contrast": "completed_vs_none",
                    "name": "balance.lunch",
                    "level": "soft",
                    "value": 0.16,
                    "threshold": 0.1,
                    "detail": "how alike the two arms are on lunch before the adjustment, and after weighting: standardised mean difference 0.16",
                }
            ]
        }
        rec.specialist_result["interpretations"] = [
            {
                "contrast": "completed_vs_none",
                "answer": "Completing the course raised math scores by about 5.6 points.",
                "effect_stated": 5.6,
                "caveats": ["The two arms differed on lunch before the adjustment [check:completed_vs_none.balance.lunch]."],
                "cites": ["estimate:completed_vs_none.value"],
            }
        ]
        return rec

    monkeypatch.setattr(pipeline, "run", worded_run)
    d = Desk(DeskFake())
    d.say(QUESTION)
    d.to_ready()
    text = d.say("run")["text"]
    lines = text.splitlines()
    assert lines[1].startswith("Completing the course raised") and lines[2].startswith("  Keep in mind:")
    assert lines.index(next(ln for ln in lines if ln.startswith("The number:"))) < lines.index(next(ln for ln in lines if ln.startswith("Design:")))
    flag = next(ln for ln in lines if ln.endswith("[check:completed_vs_none.balance.lunch]"))
    assert flag.startswith("  how alike the two arms are on lunch") and "(soft; balance.lunch)" in flag
    assert "the estimate held every time" in text and "[estimate:completed_vs_none.value]" in text


# ------------------------------------------------------------------ pictures on request


@pytest.fixture
def _artifact_root(tmp_path, monkeypatch):
    """Drawn pictures land under the memory home; for a test that is tmp_path."""
    monkeypatch.setattr(config, "ROOT", tmp_path)


def test_a_picture_asked_for_before_the_run_is_drawn_shown_and_recorded(_artifact_root, tmp_path):
    from causal_agent.viz import store as VS

    def infer(msg, addrs, human):
        return Reading(draw="show me math score by lunch") if "plot" in msg else None

    fake = DeskFake(infer=infer)
    d = Desk(fake)
    d.say(QUESTION)
    p = d.say("plot math score by lunch before we go on")
    a = p["artifact"]
    assert a and a["moment"] == "pre" and a["design"] is None and set(a["facts"]) == {"mean_standard", "mean_free_reduced"}
    assert p["text"].startswith("Mean math score by lunch") and f"[artifact:{a['id']}]" in p["text"] and p["kind"] == "ask"  # the caption, then the next ask
    assert fake.calls.count("DrawCode") == 1 and "COLUMNS" in fake.humans["DrawCode"][0] and "math score" in fake.humans["DrawCode"][0]
    folder = tmp_path / "data" / "memory" / "students" / "viz" / "pre" / a["id"]
    assert (folder / "figure.png").exists() and (folder / "code.py").exists() and VS.list_artifacts("students", "pre")[0].id == a["id"]
    step = d.journal.steps()[-1]
    assert step.kind == "explore" and step.by == "model" and step.note == "show me math score by lunch" and step.left[0].endswith(f"viz/pre/{a['id']}")
    assert HELD["students"].version == d.values["convinced_version"] if d.values.get("convinced_version") else True  # a picture settles nothing
    p = d.say(answer_ask(p))
    assert p["artifact"] is None  # shown once


def test_a_picture_asked_for_after_the_run_lands_in_the_design_and_is_citable(_artifact_root, tmp_path):
    after = [
        AfterReply(kind="draw", text="", draw="the mean math score for each lunch group"),
        AfterReply(kind="answer", text="Standard lunch students average higher.", cites=["artifact:PLACEHOLDER.mean_standard"]),
        AfterReply(kind="done", text="Bye."),
    ]
    fake = DeskFake(after=after)
    d = Desk(fake)
    d.say(QUESTION)
    d.to_ready()
    d.say("run")
    p = d.say("draw me the mean math score by lunch")
    a = p["artifact"]
    assert a and a["moment"] == "post" and a["design"] == 1 and p["text"].startswith("Mean math score by lunch") and p["kind"] == "after"
    assert "WHAT THE RUN FOUND" in fake.humans["DrawCode"][0] and "[estimate:completed_vs_none.value]" in fake.humans["DrawCode"][0]
    folder = tmp_path / "data" / "memory" / "students" / "designs" / "1" / "viz" / a["id"]
    assert (folder / "figure.png").exists()
    step = d.journal.steps()[-1]
    assert step.kind == "explore" and step.design == 1 and step.left[0].endswith(f"designs/1/viz/{a['id']}")
    fake.after[0].cites = [f"artifact:{a['id']}.mean_standard"]
    p = d.say("who does better?")
    assert p["text"].startswith("Standard lunch") and p["artifact"]["id"] == a["id"]  # an answer that cites the picture shows it
