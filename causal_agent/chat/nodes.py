"""Desk nodes. Facts: run, brief, answer, revise, requestion. Judgement, gated: turn. Interrupt: talk."""

from __future__ import annotations

import re
from typing import Literal

from langgraph.types import Command, interrupt

from causal_agent.chat import material as M
from causal_agent.chat import pipeline
from causal_agent.chat import prompts as P
from causal_agent.chat.contracts import AfterReply, Exchange, RunRecord
from causal_agent.chat.state import ChatState
from causal_agent.common.llm import structured
from causal_agent.intake.interview import nodes as I
from causal_agent.intake.knowledge import load_thresholds

TH = load_thresholds()
MAX_ATTEMPTS = 3
TOL = 0.01
NUM_RE = re.compile(r"(?<![\w:.\-])-?\d+(?:\.\d+)?(?![\w.:\-]*[a-zA-Z_])")


# ------------------------------------------------------------------ helpers


def _status_line(run: RunRecord | None) -> str:
    if run is None:
        return "no run yet"
    flags = [r["name"] for r in ((run.specialist_result.get("design") or {}).get("checks") or {}).get("results") or [] if r.get("level") != "pass"]
    eff = f"effect {M._g(run.effect)} [{M._g(run.ci_low)}, {M._g(run.ci_high)}]" if run.effect is not None else f"status {run.status}"
    return f"run {run.index} · {run.family or 'no design'} · {eff} · flags: {', '.join(flags) or 'none'}"


def _exchanges_text(state: ChatState) -> str:
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
    """A number quoted from a line's own text (a threshold, a side count, a p in a detail) is grounded by that line."""
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


def _gate(reply: AfterReply, mat: M.Material, state: ChatState) -> list[str]:
    errors: list[str] = []
    if reply.kind == "answer":
        if not reply.cites and reply.numbers:  # an address attached to a number is a cite
            reply.cites = list(dict.fromkeys(n.address for n in reply.numbers))
        bad = [c for c in reply.cites if c not in mat.addresses]
        if bad:
            errors.append(f"cites not in the material: {bad}")
        if not reply.cites:
            errors.append("an answer must cite at least one address; attach the addresses of the numbers you state, or say the material cannot answer and cite run.question")
        for n in reply.numbers:
            if n.address not in mat.addresses:
                errors.append(f"number {n.value} attached to {n.address!r}, which is not in the material")
            elif n.address in mat.numbers and abs(n.value - mat.numbers[n.address]) > max(abs(mat.numbers[n.address]), 1e-9) * TOL and not any(round(mat.numbers[n.address], d) == n.value for d in range(5)) \
                    and not _in_line(n.value, mat.by_address.get(n.address, "")):
                errors.append(f"number {n.value} does not match {n.address} = {mat.numbers[n.address]:.6g}, and does not appear in that line's text")
        stated = {round(n.value, 6) for n in reply.numbers}
        for v in _numbers_in(reply.text):
            if round(v, 6) in stated or _grounded(v, mat):
                continue
            errors.append(f"the text states {v:g}, which is in no artifact; remove it or attach its address in numbers")
    elif reply.kind == "revise":
        if not reply.claim_updates:
            errors.append("revise needs at least one claim update; if the person wants a design choice changed, answer with which claim would change it")
    elif reply.kind == "requestion":
        if not (reply.question or "").strip():
            errors.append("requestion needs the new question in full")
    return errors


# ------------------------------------------------------------------ nodes


def after_interview(state: ChatState) -> Literal["run", "__end__"]:
    return "run" if state.get("handoff_ready") else "__end__"


def run(state: ChatState) -> dict:
    runs = list(state.get("runs") or [])
    rec = pipeline.run(state["dataset"], state["question"] or "", len(runs) + 1)
    return {"runs": runs + [rec], "phase": "after"}


def brief(state: ChatState) -> dict:
    runs = state.get("runs") or []
    cur, prev = runs[-1], (runs[-2] if len(runs) > 1 else None)
    mat = M.render(cur, state.get("claims"), prev)
    return {"brief": M.brief(cur, prev, mat), "after_reply": None, "after_errors": [], "after_attempts": 0}


def talk(state: ChatState) -> Command[Literal["turn", "__end__"]]:
    runs = state.get("runs") or []
    cur = runs[-1] if runs else None
    reply = state.get("after_reply")
    text = reply.text if reply is not None else state.get("brief", "")
    payload = {"phase": "after", "text": text, "status": _status_line(cur), "ready": True, "runs": len(runs)}
    answer = str(interrupt(payload) or "").strip()
    if answer.lower() in {"quit", "exit", "done", "bye"}:
        return Command(goto="__end__")
    turn = int(state.get("after_turn", 0)) + 1
    return Command(goto="turn", update={"after_message": answer, "after_turn": turn, "after_errors": [], "after_attempts": 0})


def turn(state: ChatState) -> Command[Literal["turn", "answer", "revise", "requestion", "__end__"]]:
    runs = state.get("runs") or []
    cur = runs[-1]
    mat = M.render(cur, state.get("claims"), runs[-2] if len(runs) > 1 else None)
    errs = state.get("after_errors") or []
    errors = ("\nPREVIOUS REPLY WAS REJECTED:\n" + "\n".join(f"- {e}" for e in errs) + "\n") if errs else ""
    user = P.TURN_USER.format(material=mat.text, claims=state["claims"].render() if state.get("claims") else "(none)", kinds=I._kinds_text(),
                              exchanges=_exchanges_text(state), message=state.get("after_message", ""), errors=errors)
    out, thought = structured(AfterReply, P.TURN_SYSTEM, user, node=f"turn:{state.get('after_turn', 0)}")
    gate = _gate(out, mat, state)
    attempts = int(state.get("after_attempts", 0)) + 1
    if gate and attempts < MAX_ATTEMPTS:
        return Command(goto="turn", update={"after_errors": gate, "after_attempts": attempts, "debug": [thought]})
    if gate:
        out = AfterReply(kind="answer", text="I cannot ground that in the run's artifacts: " + "; ".join(gate) + ". Ask about what the run left behind, or tell me a claim to change.", cites=["run.question"])
        gate = [f"fell back after {attempts} tries: " + "; ".join(gate)]
    ex = Exchange(turn=int(state.get("after_turn", 0)), user=state.get("after_message", ""), assistant=out.text, kind=out.kind)
    goto = {"answer": "answer", "revise": "revise", "requestion": "requestion", "done": "__end__"}[out.kind]
    return Command(goto=goto, update={"after_reply": out, "after_errors": gate, "after_attempts": 0, "exchanges": [ex], "debug": [thought]})


def answer(state: ChatState) -> dict:
    return {}


def revise(state: ChatState) -> Command[Literal["interview", "talk"]]:
    reply = state["after_reply"]
    table = state["claims"].model_copy(deep=True)
    df, _ = I._data(state, table)
    turn_n = int(state.get("turn", 0)) + 1
    allowed = {f"user:turn:{turn_n}"}
    rejected = []
    for up in reply.claim_updates:
        up = up.model_copy(update={"cites": [f"user:turn:{turn_n}"]})
        err = I._apply(table, up, df, allowed, turn_n)
        if err:
            rejected.append(err)
    if rejected and all(rejected):
        note = "I could not apply that change: " + "; ".join(rejected)
        return Command(goto="talk", update={"after_reply": reply.model_copy(update={"text": note, "kind": "answer"})})
    msg = state.get("after_message", "")
    return Command(goto="interview", update={
        "claims": table, "turn": turn_n, "last_message": msg, "last_source": f"user:turn:{turn_n}", "phase": "before",
        "messages": [{"role": "assistant", "turn": turn_n - 1, "text": reply.text}, {"role": "user", "turn": turn_n, "text": msg}],
        "extract_errors": [], "extract_attempts": 0, "respond_errors": [], "respond_attempts": 0, "handoff_ready": False,
    })


def requestion(state: ChatState) -> dict:
    return {"question": state["after_reply"].question}
