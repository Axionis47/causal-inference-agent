"""The chat after a run. Facts: brief, answer, revise's gate, requestion. Judgement, gated: turn. Interrupt: talk.
Everything is answered from the run's artifacts and the memory; a change goes through the gate and back to the checks."""

from __future__ import annotations

import re
from typing import Literal

from langgraph.types import Command, interrupt

from causal_agent.common.contracts import Said
from causal_agent.common.llm import structured
from causal_agent.desk import material as M
from causal_agent.desk import pipeline
from causal_agent.desk.contracts import AfterReply, Exchange, RunRecord
from causal_agent.desk.nodes import frame as F
from causal_agent.desk.nodes.journey import CAT, QUIT_WORDS, kinds_text
from causal_agent.desk.prompts import journey as P
from causal_agent.desk.state import DeskState
from causal_agent.memory import ops, store

MAX_ATTEMPTS = 3
TOL = 0.01
NUM_RE = re.compile(r"(?<![\w:.\-])-?\d+(?:\.\d+)?(?![\w.:\-]*[a-zA-Z_])")
DONE_WORDS = {"done", "bye", "finished"} | QUIT_WORDS


# ------------------------------------------------------------------ helpers


def _status_line(run: RunRecord | None) -> str:
    if run is None:
        return "no run yet"
    flags = [r["name"] for r in ((run.specialist_result.get("design") or {}).get("checks") or {}).get("results") or [] if r.get("level") != "pass"]
    eff = f"effect {M._g(run.effect)} [{M._g(run.ci_low)}, {M._g(run.ci_high)}]" if run.effect is not None else f"status {run.status}"
    return f"run {run.index} · {run.family or 'no design'} · {eff} · flags: {', '.join(flags) or 'none'}"


def _exchanges_text(state: DeskState) -> str:
    ex = state.get("exchanges") or []
    return "\n".join(f"[{e.turn}] person: {e.user}\n[{e.turn}] you ({e.kind}): {e.assistant[:600]}" for e in ex[-6:]) or "(none yet)"


def _numbers_in(text: str) -> list[float]:
    out = []
    for tok in NUM_RE.findall(text):
        try:
            v = float(tok)
        except ValueError:
            continue
        if abs(v) >= 10 or "." in tok:
            out.append(v)
    return out


def _in_line(v: float, line: str) -> bool:
    for tok in NUM_RE.findall(line):
        try:
            x = float(tok)
        except ValueError:
            continue
        if abs(v - x) <= max(abs(x), 1e-9) * TOL or any(round(x, d) == v for d in range(5)):
            return True
    return False


def _grounded(v: float, mat: M.Material) -> bool:
    for x in mat.numbers.values():
        if abs(v - x) <= max(abs(x), 1e-9) * TOL:
            return True
        for d in (0, 1, 2, 3, 4):
            if round(x, d) == v:
                return True
    return f"{v:g}" in mat.text or str(v) in mat.text


def _gate(reply: AfterReply, mat: M.Material) -> list[str]:
    errors: list[str] = []
    if reply.kind == "answer":
        if not reply.cites and reply.figure:
            reply.cites = [reply.figure]
        if not reply.cites and reply.numbers:
            reply.cites = list(dict.fromkeys(n.address for n in reply.numbers))
        bad = [c for c in reply.cites if c not in mat.addresses]
        if bad:
            errors.append(f"cites not in the material: {bad}")
        if not reply.cites:
            errors.append(
                "an answer must cite at least one address; attach the addresses of the numbers you state, or say the material cannot answer and cite run.question"
            )
        for n in reply.numbers:
            if n.address not in mat.addresses:
                errors.append(f"number {n.value} attached to {n.address!r}, which is not in the material")
            elif (
                n.address in mat.numbers
                and abs(n.value - mat.numbers[n.address]) > max(abs(mat.numbers[n.address]), 1e-9) * TOL
                and not any(round(mat.numbers[n.address], d) == n.value for d in range(5))
                and not _in_line(n.value, mat.by_address.get(n.address, ""))
            ):
                errors.append(f"number {n.value} does not match {n.address} = {mat.numbers[n.address]:.6g}, and does not appear in that line's text")
        stated = {round(n.value, 6) for n in reply.numbers}
        for v in _numbers_in(reply.text):
            if round(v, 6) in stated or _grounded(v, mat):
                continue
            errors.append(f"the text states {v:g}, which is in no artifact; remove it or attach its address in numbers")
    if reply.figure and reply.figure not in mat.addresses:
        errors.append(f"figure {reply.figure!r} is not in the material; name one of the figure: addresses or leave it empty")
    if reply.kind in ("revise", "what_if"):
        if not reply.updates:
            errors.append(f"{reply.kind} needs at least one field update; if the person wants a design choice changed, answer with which field would change it")
    elif reply.kind == "requestion":
        if not (reply.question or "").strip():
            errors.append("requestion needs the new question in full")
    return errors


# ------------------------------------------------------------------ nodes


def brief(state: DeskState) -> dict:
    runs = list(state.get("runs") or [])
    cur, prev = runs[-1], (runs[-2] if len(runs) > 1 else None)
    memory = F.memory_of(state)
    if prev is not None:  # then and now: which fields differed between the two designs
        cur.differs = design_differences(prev.design_dir, cur.design_dir)
        pipeline.save_record(cur)
    mat = M.render(cur, memory, prev)
    text = M.brief(cur, prev, mat)
    if cur.what_if:
        text += "\nThis was a what-if: nothing you told me changed. The copy differed in " + ", ".join(f"[{a}] = {v}" for a, v in cur.what_if.items()) + "."
    elif cur.differs:
        text += "\nWhat differed from the design before: " + ", ".join(f"[{a}]" for a in cur.differs) + "."
    return {"brief": text, "after_reply": None, "after_errors": [], "after_attempts": 0, "reply": text, "runs": runs, "fork": None, "what_if": {}}


def design_differences(before_dir: str | None, after_dir: str | None) -> list[str]:
    """The addresses whose value or status differs between two design snapshots."""
    import json
    from pathlib import Path

    try:
        a = json.loads((Path(before_dir) / "memory.json").read_text())["fields"]
        b = json.loads((Path(after_dir) / "memory.json").read_text())["fields"]
    except Exception:
        return []
    out = []
    for addr in sorted(set(a) | set(b)):
        fa, fb = a.get(addr) or {}, b.get(addr) or {}
        if (fa.get("value"), fa.get("status")) != (fb.get("value"), fb.get("status")):
            out.append(addr)
    return out


def talk(state: DeskState) -> Command[Literal["turn", "__end__"]]:
    runs = state.get("runs") or []
    cur = runs[-1] if runs else None
    reply = state.get("after_reply")
    text = reply.text if reply is not None else state.get("brief", "")
    figure = None
    if reply is not None and reply.figure and cur is not None:
        figure = next((f for f in cur.figures if f"figure:{f.get('id')}" == reply.figure), None)
    elif reply is None and cur is not None and cur.figures:
        figure = cur.figures[min(1, len(cur.figures) - 1)]  # the run's own figure, or the ready-moment one when the run made none
    payload = {
        "phase": "after",
        "kind": "after",
        "text": text,
        "status": _status_line(cur),
        "ready": True,
        "runs": len(runs),
        "open": [],
        "ask": None,
        "figure": figure,
    }
    answer = str(interrupt(payload) or "").strip()
    if answer.lower() in DONE_WORDS:
        return Command(goto="__end__")
    memory = F.memory_of(state)
    turn = int(state.get("turn") or 0) + 1
    if not any(s.turn == turn for s in memory.said):
        memory.said.append(Said(turn=turn, about="after", text=answer))
        store.save(memory)
    return Command(goto="turn", update={"message": answer, "turn": turn, "after_errors": [], "after_attempts": 0})


def turn(state: DeskState) -> Command[Literal["turn", "answer", "revise", "what_if", "requestion", "__end__"]]:
    runs = state.get("runs") or []
    cur = runs[-1]
    memory = F.memory_of(state)
    mat = M.render(cur, memory, runs[-2] if len(runs) > 1 else None)
    errs = state.get("after_errors") or []
    errors = ("\nPREVIOUS REPLY WAS REJECTED:\n" + "\n".join(f"- {e}" for e in errs) + "\n") if errs else ""
    user = P.TURN_USER.format(
        material=mat.text,
        memory=memory.render() or "(nothing known)",
        kinds=kinds_text(),
        exchanges=_exchanges_text(state),
        message=state.get("message", ""),
        errors=errors,
    )
    out, thought = structured(AfterReply, P.TURN_SYSTEM, user, node=f"turn:{state.get('turn', 0)}")
    gate = _gate(out, mat)
    attempts = int(state.get("after_attempts", 0)) + 1
    if gate and attempts < MAX_ATTEMPTS:
        return Command(goto="turn", update={"after_errors": gate, "after_attempts": attempts, "debug": [thought]})
    if gate:
        out = AfterReply(
            kind="answer",
            text="I cannot ground that in the run's artifacts: " + "; ".join(gate) + ". Ask about what the run left behind, or tell me something to change.",
            cites=["run.question"],
        )
        gate = [f"fell back after {attempts} tries: " + "; ".join(gate)]
    ex = Exchange(turn=int(state.get("turn", 0)), user=state.get("message", ""), assistant=out.text, kind=out.kind)
    goto = {"answer": "answer", "revise": "revise", "what_if": "what_if", "requestion": "requestion", "done": "__end__"}[out.kind]
    return Command(goto=goto, update={"after_reply": out, "after_errors": gate, "after_attempts": 0, "exchanges": [ex], "debug": [thought]})


def answer(state: DeskState) -> dict:
    return {}


def revise(state: DeskState) -> Command[Literal["check", "talk"]]:
    """The person's words go through the same gate as before the run; what is accepted sends the journey back to the checks."""
    reply = state["after_reply"]
    memory = F.memory_of(state)
    turn = int(state.get("turn") or 0)
    src = f"user:turn:{turn}"
    updates = [
        ops.Update(address=u.address, value=u.value, status="confirmed", source=src, said=u.said or state.get("message", "")[:200], reason=u.reason)
        for u in reply.updates
    ]
    before = {a: (f.value, f.status) for a, f in memory.fields.items()}
    rejected = ops.apply(memory, updates, CAT)
    settled = [a for a, f in memory.fields.items() if before.get(a) != (f.value, f.status)]
    store.save(memory)
    if not settled:
        note = "I could not apply that change: " + "; ".join(rejected)
        return Command(goto="talk", update={"after_reply": reply.model_copy(update={"text": note, "kind": "answer"})})
    return Command(
        goto="check", update={"phase": "before", "settled_now": settled, "run_requested": False, "infer_errors": [], "infer_attempts": 0, "handoff": None}
    )


def what_if(state: DeskState) -> Command[Literal["fit", "talk"]]:
    """A supposition: the memory is copied, the supposed fields written on the copy through the same gate, and the copy is routed
    and run as the next design. What is known does not change."""
    reply = state["after_reply"]
    memory = F.memory_of(state)
    fork = memory.fork()
    turn = int(state.get("turn") or 0)
    src = f"user:turn:{turn}"
    updates = [
        ops.Update(address=u.address, value=u.value, status="confirmed", source=src, said=u.said or state.get("message", "")[:200], reason=u.reason)
        for u in reply.updates
    ]
    before = {a: (f.value, f.status) for a, f in fork.fields.items()}
    rejected = ops.apply(fork, updates, CAT)
    changed = {a: f.value for a, f in fork.fields.items() if before.get(a) != (f.value, f.status)}
    if not changed:
        note = "I could not suppose that: " + "; ".join(rejected)
        return Command(goto="talk", update={"after_reply": reply.model_copy(update={"text": note, "kind": "answer"})})
    return Command(
        goto="fit", update={"fork": fork, "what_if": {a: str(v) for a, v in changed.items()}, "handoff": None, "gate_errors": [], "decide_attempts": 0}
    )


def requestion(state: DeskState) -> dict:
    q = state["after_reply"].question or ""
    return {"question": q, "message": q, "phase": "before", "handoff": None, "invalid": None, "prefilter_votes": [], "oriented": False, "focus": []}
