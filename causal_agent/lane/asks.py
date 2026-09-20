"""One question back to the desk, the same shape in every lane. The lane stops at `feasibility` with stage 'ask'; the
desk asks the question, takes the answer through its gate, and runs the lane again on the memory as it then stands."""

from __future__ import annotations

from langgraph.types import Command

from causal_agent.common.contracts import Feasibility, LaneAsk


def ask_back(stage: str, ask: LaneAsk, reason: str, facts: list[str] | None = None, extra: dict | None = None) -> Command:
    ask = ask.model_copy(update={"stage": stage}) if not ask.stage else ask
    f = Feasibility(stage="ask", reason=reason, facts=list(facts or []), what_would_fix=f"an answer to [{ask.address}]")
    return Command(goto="feasibility", update={"feasibility": f, "ask": ask, **(extra or {})})


def status_of(state: dict) -> str:
    if state.get("ask"):
        return "ask"
    return "infeasible" if state.get("feasibility") else "done"


def ask_dict(ask) -> dict | None:
    if ask is None:
        return None
    return ask.model_dump() if hasattr(ask, "model_dump") else dict(ask)
