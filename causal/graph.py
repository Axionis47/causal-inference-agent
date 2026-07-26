"""The run as a graph, so it can stop at the design gate and resume later.

This is the only place LangGraph appears, and it is here for one reason:
checkpointing. A run reaches the design gate, parks, and waits for a person who
may answer in ten seconds or tomorrow morning. Hand-rolling that means writing
paused state to disk, rebuilding it on resume, and re-arming the rest of the
pipeline. `interrupt()` plus a SqliteSaver does it in a few lines and survives
a server restart.

    read ──▶ menu ──▶ ⟨interrupt: you choose⟩ ──▶ estimate ──▶ narrate

Everywhere else a node would only wrap a function call so the graph could call
the function, which is cost without benefit. Those steps stay plain Python.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import lanes as lane_module
from .design import options
from .intake import read_question
from .profile import profile
from .report import claim_strength, headline, narrate

CHECKPOINTS = Path(__file__).parent.parent / "jobs.sqlite"

LANES = {
    "observational": lane_module.observational,
    "matching": lane_module.matching,
    "did": lane_module.did,
    "rdd": lane_module.rdd,
    "iv": lane_module.iv,
    "survival": lane_module.survival,
    "mediation": lane_module.mediation,
    "time_series": lane_module.time_series,
}


class State(TypedDict, total=False):
    # inputs
    csv_path: str
    question: str
    context: str
    # produced as it goes
    columns: list[dict]
    n_rows: int
    intake: dict
    menu: list[dict]
    choice: dict  # {"lane": str, "kwargs": {...}} — supplied on resume
    estimate: dict
    strength: str
    headline: str
    narrative: str
    error: str
    events: list[dict]


def _frame(state: State) -> pd.DataFrame:
    return pd.read_csv(state["csv_path"])


def _event(state: State, kind: str, **fields) -> list[dict]:
    """Append to the event log the SSE endpoint replays.

    Events are stored on the graph state rather than pushed to a queue, so a
    client that reconnects sees everything it missed. The stream is a view of
    this list, not the only copy of it.
    """
    return [*state.get("events", []), {"event": kind, **fields}]


def node_read(state: State) -> dict:
    df = _frame(state)
    p = profile(df)
    events = _event(state, "stage_started", stage="read")
    intake = read_question(state["question"], state.get("context", ""), p)
    if not intake.usable:
        return {
            "columns": [c.__dict__ for c in p.columns],
            "n_rows": p.n_rows,
            "error": intake.problem,
            "events": [*events, {"event": "failed", "reason": intake.problem}],
        }
    return {
        "columns": [c.__dict__ for c in p.columns],
        "n_rows": p.n_rows,
        "intake": intake.__dict__ | {"exposure": intake.exposure},
        "events": [*events, {"event": "stage_done", "stage": "read",
                             "detail": f"{intake.exposure} -> {intake.outcome}"}],
    }


def node_menu(state: State) -> dict:
    if state.get("error"):
        return {}
    df = _frame(state)
    p = profile(df)
    menu = options(df, p, treatment=state["intake"].get("treatment"),
                   outcome=state["intake"]["outcome"])
    return {
        "menu": [o.__dict__ for o in menu],
        "events": _event(state, "stage_done", stage="menu",
                         detail=f"{sum(o.available for o in menu)} designs available"),
    }


def node_gate(state: State) -> dict:
    """Park here. The person picks the design; nothing guesses it.

    `interrupt` raises out of the graph run and the checkpointer stores
    everything above. The resume call supplies the choice.
    """
    if state.get("error"):
        return {}
    choice = interrupt({
        "question": "Which design should this run use?",
        "menu": state["menu"],
        "intake": state["intake"],
    })
    return {
        "choice": choice,
        "events": _event(state, "stage_done", stage="gate",
                         detail=f"you chose {choice.get('lane')}"),
    }


def node_estimate(state: State) -> dict:
    if state.get("error"):
        return {}
    lane = state["choice"]["lane"]
    events = _event(state, "stage_started", stage="estimate", detail=lane)
    if lane not in LANES:
        return {"error": f"no lane called '{lane}'",
                "events": [*events, {"event": "failed", "reason": f"no lane '{lane}'"}]}
    try:
        est = LANES[lane](_frame(state), **state["choice"].get("kwargs", {}))
    except Exception as exc:  # a lane refusing is a result, not a crash
        return {"error": str(exc),
                "events": [*events, {"event": "failed", "reason": str(exc)}]}
    intake = state["intake"]
    return {
        "estimate": est.__dict__,
        "strength": claim_strength(lane, est),
        "headline": headline(lane, est, intake.get("exposure", ""), intake["outcome"]),
        "events": [*events, {"event": "stage_done", "stage": "estimate",
                             "detail": str(est)}],
    }


def node_narrate(state: State) -> dict:
    if state.get("error") or not state.get("estimate"):
        return {"events": _event(state, "failed", reason=state.get("error", "no estimate"))}
    from .estimate import Estimate

    est = Estimate(**state["estimate"])
    intake = state["intake"]
    text = narrate(state["choice"]["lane"], est,
                   intake.get("exposure", ""), intake["outcome"])
    return {"narrative": text,
            "events": _event(state, "completed", detail=state["headline"])}


def build():
    """The graph, compiled with a checkpointer so a parked run survives restart."""
    g = StateGraph(State)
    g.add_node("read", node_read)
    g.add_node("menu", node_menu)
    g.add_node("gate", node_gate)
    g.add_node("estimate", node_estimate)
    g.add_node("narrate", node_narrate)
    g.add_edge(START, "read")
    g.add_edge("read", "menu")
    g.add_edge("menu", "gate")
    g.add_edge("gate", "estimate")
    g.add_edge("estimate", "narrate")
    g.add_edge("narrate", END)

    conn = sqlite3.connect(CHECKPOINTS, check_same_thread=False)
    return g.compile(checkpointer=SqliteSaver(conn))
