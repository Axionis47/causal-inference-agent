"""The hand-off and the run: gate the decision, project the pack, run the lane in its own process, and let the lane ask
back once."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import Command

from causal_agent.desk import pipeline
from causal_agent.desk.contracts import Ask, RunRecord
from causal_agent.desk.nodes import decide as D
from causal_agent.desk.nodes import frame as F
from causal_agent.desk.nodes.shared import CAT
from causal_agent.desk.state import Context, DeskState
from causal_agent.families import registry as R
from causal_agent.memory import store
from causal_agent.memory.records import COLUMN_KIND, Memory

# ------------------------------------------------------------------ the hand-off and the run


def gate(state: DeskState, runtime: Runtime[Context]) -> Command[Literal["decide", "handoff"]]:
    """The routing gate; a choice that fails its checks three times still reaches handoff, which records the honest stop."""
    cmd = D.gate(state, runtime)
    if cmd.goto == "__end__":
        return Command(update=cmd.update, goto="handoff")
    return cmd


def handoff(state: DeskState, runtime: Runtime[Context]) -> dict:
    """The pack from the memory, as design n: the memory snapshot and the pack written under designs/<n>/."""
    out = D.handoff(state, runtime)
    h = out.get("handoff")
    memory = F.memory_of(state)
    n = len(state.get("runs") or []) + 1
    d = store.snapshot(memory, n)
    if h is not None:
        h.design_id = n
        (d / "handoff.json").write_text(h.model_dump_json(indent=2))
    (d / "record.md").write_text(out.get("decision_record") or "")
    if state.get("frame") is not None:
        (d / "frame.json").write_text(state["frame"].model_dump_json(indent=2))
    if state.get("decision") is not None:
        (d / "decision.json").write_text(state["decision"].model_dump_json(indent=2))
    return {**out, "design_dir": str(d)}


def _decision(state: DeskState) -> dict:
    d, h = state.get("decision"), state.get("handoff")
    out: dict = {}
    if d is not None:
        out = {"chosen": d.chosen, "chosen_assumption": d.chosen_assumption, "why": d.why_over_alternatives, "over": {r.family: r.reason for r in d.rejected}}
    if h is not None:
        out.setdefault("chosen", h.family)
        out.setdefault("chosen_assumption", h.chosen_assumption)
    return out


def run(state: DeskState) -> dict:
    runs = list(state.get("runs") or [])
    n = len(runs) + 1
    h = state.get("handoff")
    question = state.get("question") or ""
    if h is None:
        rec = RunRecord(
            index=n,
            dataset=state["dataset"],
            question=question,
            status="no_handoff",
            decision=_decision(state),
            decision_record=state.get("decision_record") or "",
            design_dir=state.get("design_dir"),
        )
    else:
        rec = pipeline.run(
            Path(state["design_dir"]) / "handoff.json",
            n,
            state["dataset"],
            question,
            decision=_decision(state),
            decision_record=state.get("decision_record") or "",
        )
    rec.what_if = dict(state.get("what_if") or {})
    rec.figures = _figures(state, rec)
    if state.get("design_dir"):
        import json

        (Path(state["design_dir"]) / "figures.json").write_text(json.dumps(rec.figures, indent=2, default=str))
    return {"runs": runs + [rec], "phase": "after"}


def _figures(state: DeskState, rec: RunRecord) -> list[dict]:
    """The ready-moment figure first, then what the lane wrote and checked under its run dir; the post-viz fallback for a lane
    that wrote none."""
    import json

    from causal_agent.viz import postviz

    ready = [{**state["figure"], "moment": "ready"}] if state.get("figure") else []
    lane: list[dict] = []
    p = Path(rec.run_dir) / "figures.json" if rec.run_dir else None
    if p is not None and p.exists():
        try:
            lane = [f for f in json.loads(p.read_text()) if isinstance(f, dict) and f.get("id")]
        except Exception:
            lane = []
    if not lane:
        prefix = R.REGISTRY[rec.family].refutation_prefix if rec.family in R.REGISTRY else "placebo"
        lane = [f.model_dump() for f in postviz.figures(rec, prefix)]
    return ready + lane


# ------------------------------------------------------------------ the lane asks back


def _asked(rec: RunRecord | None) -> str | None:
    """The address a run asked back about, if it did."""
    if rec is None or rec.status != "ask":
        return None
    return ((rec.specialist_result or {}).get("ask") or {}).get("address") or None


def after_run(state: DeskState) -> Literal["ask_back", "brief"]:
    """A lane that asks back is asked once per address: the same question from the run before goes to the brief, which
    says the lane still asks it."""
    runs = state.get("runs") or []
    address = _asked(runs[-1]) if runs else None
    if not address:
        return "brief"
    if len(runs) > 1 and _asked(runs[-2]) == address:
        return "brief"
    return "ask_back"


def ask_back(state: DeskState) -> dict:
    """The lane could not go on without one more thing from the person: ask it, as the desk asks everything else, and once
    answered the journey runs again on the memory as it then stands. The lane's own options and evidence are used when it
    gives them; otherwise the catalogue's field type says what the legal answers are."""
    rec = (state.get("runs") or [])[-1]
    q = rec.specialist_result.get("ask") or {}
    address = q.get("address") or ""
    kind_name = Memory.parse(address)[1] if address.startswith("claim:") else COLUMN_KIND
    field = Memory.parse(address)[2] or ""
    kind = CAT.kinds.get(kind_name)
    options = [str(o) for o in q.get("options") or []]
    if not options:
        options = (
            [str(o) for o in kind.fields[field].options]
            if kind and field in kind.fields and kind.fields[field].type == "choice"
            else (["yes", "no"] if kind and field in kind.fields and kind.fields[field].type == "bool" else [])
        )
    a = Ask(
        addresses=[address],
        kind="choose" if options else "open",
        text=q.get("question") or "",
        options=options,
        because=[rec.family] if rec.family else [],
        evidence=[str(e) for e in q.get("evidence") or []],
        from_lane=True,
    )
    reason = (rec.specialist_result.get("feasibility") or {}).get("reason") or "it needs one more thing"
    because = f"{q['because']}\n" if q.get("because") else ""
    text = f"The analysis stopped before estimating: {reason}.\n\n{because}{a.text}"
    return {"ask": a, "reply": text, "phase": "before", "run_requested": False, "figure": None, "fork": None, "what_if": {}, "handoff": None}
