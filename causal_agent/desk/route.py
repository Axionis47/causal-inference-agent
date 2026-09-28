"""The routing alone: from a question and a memory to a hand-off, without the interview. The same node functions the desk
runs, called in order on a plain dict; the caller runs the lane.

    load ─ mine ─(prefilter × N, wide only)─ frame ─ fit ─ decide ─ gate ─ handoff
"""

from __future__ import annotations

from pydantic import BaseModel

from causal_agent.common.contracts import FamilyDecision, FamilyVerdict, Handoff, QuestionFrame, Thought
from causal_agent.desk.nodes import decide as D
from causal_agent.desk.nodes import frame as F

APPENDED = {"prefilter_votes", "debug"}  # the RouteState keys with an add reducer; every other key replaces


class RouteResult(BaseModel):
    handoff: Handoff | None
    decision: FamilyDecision | None
    frame: QuestionFrame | None
    family_verdicts: list[FamilyVerdict]
    probes: list  # ProbeResult
    decision_record: str
    debug: list[Thought]


def _merge(state: dict, update: dict | None) -> None:
    """Fold a node's return into the state as the graph would."""
    for k, v in (update or {}).items():
        state[k] = state.get(k, []) + v if k in APPENDED else v


def route(question: str, dataset: str) -> RouteResult:
    """From a question and a memory to a hand-off, without the interview: mine the note once when the memory is bare, skim the
    columns when the table is wide, read the question, fit, decide, gate (up to three tries), and build the pack. The same node
    functions the desk runs, called in order. After three failed gates there is no hand-off; the decision record is still written."""
    state: dict = {"question": question, "dataset": dataset}
    _merge(state, F.load(state))
    _merge(state, F.mine(state))
    sends = F.fan_out_prefilter(state)
    if isinstance(sends, list):
        for send in sends:
            _merge(state, F.prefilter(send.arg))
    _merge(state, F.frame(state))
    _merge(state, D.fit(state, None))
    while True:
        _merge(state, D.decide(state, None))
        cmd = D.gate(state, None)
        _merge(state, cmd.update)
        if cmd.goto != "decide":
            break
    _merge(state, D.handoff(state, None))
    return RouteResult(
        handoff=state["handoff"] if cmd.goto == "handoff" else None,
        decision=state.get("decision"),
        frame=state.get("frame"),
        family_verdicts=state.get("family_verdicts", []),
        probes=state.get("probes", []),
        decision_record=state.get("decision_record", ""),
        debug=state.get("debug", []),
    )
