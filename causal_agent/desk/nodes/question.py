"""The first thing asked is the causal question, validated against the file: load, ask, read, and read again."""

from __future__ import annotations

from typing import Literal

from langgraph.types import Command, interrupt

from causal_agent.common.contracts import QuestionFrame
from causal_agent.desk.nodes import frame as F
from causal_agent.desk.nodes.shared import CAT, QUIT_WORDS, _remember, _writer
from causal_agent.desk.state import DeskState
from causal_agent.memory import ops, store
from causal_agent.memory.records import Memory

# ------------------------------------------------------------------ load and the question


def load(state: DeskState) -> dict:
    memory = F.memory_of(state)  # raises for an unknown dataset
    # the turn counter continues from the last word the memory holds, so a new conversation never reuses a number and loses a turn
    return {
        "turn": max(int(state.get("turn") or 0), max((s.turn for s in memory.said), default=0)),
        "phase": "before",
        "invalid": None,
        "frame_attempts": 0,
        "findings": [],
        "refutations": {},
        "settled_now": [],
        "ask": None,
        "oriented": False,
        "note": "",
        "focus": [],
        "desk_question": None,
        "explained": None,
        "explain_errors": [],
        "explain_attempts": 0,
        "open": [],
        "status": None,
        "infer_errors": [],
        "infer_attempts": 0,
        "run_requested": False,
        "ready": False,
        "runs": list(state.get("runs") or []),
        "prefilter_votes": [],
        "family_verdicts": [],
        "probes": [],
        "fit_status": None,
        "gate_errors": [],
        "decide_attempts": 0,
        "debug": [],
        "after_reply": None,
        "after_errors": [],
        "after_attempts": 0,
        "brief": "",
    }


def ask_question(state: DeskState) -> Command[Literal["mine", "__end__"]]:
    """The first turn, and every turn after a question that did not pass: the causal question, nothing before it."""
    memory = F.memory_of(state)
    invalid = state.get("invalid")
    if invalid:
        text = invalid
    else:
        cols = ", ".join(c.name for c in memory.columns.values())
        rows = memory.facts.get("rows")
        text = (
            f"What is the causal question you want answered from this file? Say what changed and what it might have affected. "
            f"The file has {rows} rows and these columns: {cols}."
        )
    payload = {"phase": "before", "kind": "question", "text": text, "status": "", "ready": False, "open": [], "ask": None}
    answer = str(interrupt(payload) or "").strip()
    if answer.lower() in QUIT_WORDS:
        return Command(goto="__end__")
    turn = int(state.get("turn") or 0) + 1
    _remember(memory, turn, "question", answer)
    store.save(memory)
    return Command(goto="mine", update={"question": answer, "message": answer, "turn": turn, "invalid": None, "prefilter_votes": []})


def validate(frame: QuestionFrame, memory: Memory) -> list[str]:
    """Whether the question, as read, is one this file can answer. Code over the frame and the file's facts."""
    out: list[str] = []
    if frame.intent != "effect_of_change":
        out.append(
            {
                "driver_search": "it asks what drives an outcome rather than what one change did to it",
                "root_cause": "it asks why something happened rather than what one change did to an outcome",
                "not_causal": "it does not ask what a change did to an outcome",
            }[frame.intent]
        )
    oc = memory.column(frame.outcome) if frame.outcome else None
    if oc is None:
        out.append("I could not match the outcome to a column in this file")
    elif oc.facts.constant:
        out.append(f"'{oc.name}' takes one value in this file, so nothing can have moved it")
    cc = memory.column(frame.cause) if frame.cause else None
    if frame.intent == "effect_of_change":
        if cc is None:
            out.append("I could not find a column that records who got the change, or when; name it, or say which column tells the rows apart")
        elif cc.facts.constant:
            out.append(f"every row has the same value of '{cc.name}', so there is nobody without the change to compare against")
        elif oc is not None and cc.key == oc.key:
            out.append("the change and the outcome are the same column")
    return out


def read_question(state: DeskState) -> Command[Literal["ask_question", "check"]]:
    """One judgement reads the question against the memory; code decides whether it passes."""
    memory = F.memory_of(state)
    out = F.frame(state)
    fr: QuestionFrame = out["frame"]
    problems = validate(fr, memory)
    attempts = int(state.get("frame_attempts") or 0) + 1
    _writer()({"read_question": {"intent": fr.intent, "outcome": fr.outcome, "cause": fr.cause, "problems": problems}})
    if problems:
        text = (
            "That is not yet a question I can answer from this file: "
            + "; ".join(problems)
            + ". Ask it again, naming the change and the outcome as they appear in the columns."
        )
        return Command(goto="ask_question", update={"frame": fr, "invalid": text, "frame_attempts": attempts, "debug": out["debug"]})
    note = ""
    old = memory.value("claim:assignment.treatment_column")
    if (
        state.get("runs")
        and fr.cause
        and old
        and memory.column(old) is not None
        and memory.column(fr.cause) is not None
        and memory.column(old).key != memory.column(fr.cause).key
    ):
        # a new change: everything settled relative to the old one is asked again; what a column is carries over
        dropped = ops.forget_change(memory)
        note = f"The change is not the one before ({old}), so what was settled relative to it is asked again ({len(dropped)} fields); what each column is carries over. "
    if fr.cause and memory.value("claim:assignment.treatment_column") is None:  # the frame's reading, as a draft the person confirms
        ops.apply(memory, [ops.Update(address="claim:assignment.treatment_column", value=fr.cause, status="drafted", source="code:frame")], CAT)
    store.save(memory)
    return Command(goto="check", update={"frame": fr, "invalid": None, "frame_attempts": attempts, "debug": out["debug"], "settled_now": [], "note": note})
