"""The chat after a run. Facts: brief, answer, revise's gate, requestion, draw_after. Judgement, gated: turn. Interrupt: talk.
Everything is answered from the run's artifacts and the memory; a change goes through the gate and back to the checks; a
picture is drawn by the tool from the file and the pack, and cited like any other artifact."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from langgraph.types import Command, interrupt

from causal_agent.common.contracts import Handoff, RunRecord, Said
from causal_agent.common.llm import structured
from causal_agent.desk import material as M
from causal_agent.desk import pipeline
from causal_agent.desk.contracts import AfterReply, Exchange
from causal_agent.desk.nodes import frame as F
from causal_agent.desk.nodes.interview import draw_context
from causal_agent.desk.nodes.journey import CAT, QUIT_WORDS, kinds_text
from causal_agent.desk.nodes.shared import _csv_path, design_now, journal_of, record
from causal_agent.desk.prompts import journey as P
from causal_agent.desk.state import DeskState
from causal_agent.memory import ops, store
from causal_agent.viz import draw as VD
from causal_agent.viz import store as VS

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
    elif reply.kind == "draw":
        if not (reply.draw or "").strip():
            errors.append("draw needs what to draw, in the person's words")
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
    run_step = journal_of(state).last("run")
    record(
        state,
        "brief",
        by="code",
        memory=memory,
        design=int(cur.index),
        read=[run_step.address] if run_step else [],
        left=[str(Path(cur.design_dir) / "record.json")] if prev is not None and cur.design_dir else [],
        note=text.splitlines()[0] if text else "",
    )
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
        figure = cur.figures[0]  # the brief shows the run's first figure
    artifact = state.get("artifact")
    if artifact is None and reply is not None and cur is not None:  # an answer that cites a drawn picture shows it
        cited = [c.split(".")[0][len("artifact:") :] for c in reply.cites if c.startswith("artifact:")]
        if cited:
            pool = VS.list_artifacts(cur.dataset, "pre") + VS.list_artifacts(cur.dataset, "post", cur.index)
            found = next((a for a in pool if a.id == cited[0]), None)
            artifact = found.model_dump() if found else None
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
        "artifact": artifact,
    }
    answer = str(interrupt(payload) or "").strip()
    if answer.lower() in DONE_WORDS:
        return Command(goto="__end__")
    memory = F.memory_of(state)
    turn = int(state.get("turn") or 0) + 1
    if not any(s.turn == turn for s in memory.said):
        memory.said.append(Said(turn=turn, about="after", text=answer))
        store.save(memory)
    return Command(goto="turn", update={"message": answer, "turn": turn, "after_errors": [], "after_attempts": 0, "artifact": None})


def turn(state: DeskState) -> Command[Literal["turn", "answer", "revise", "what_if", "requestion", "draw_after", "__end__"]]:
    runs = state.get("runs") or []
    cur = runs[-1]
    memory = F.memory_of(state)
    mat = M.render(cur, memory, runs[-2] if len(runs) > 1 else None, steps=journal_of(state).steps())
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
    goto = {"answer": "answer", "revise": "revise", "what_if": "what_if", "requestion": "requestion", "draw": "draw_after", "done": "__end__"}[out.kind]
    return Command(goto=goto, update={"after_reply": out, "after_errors": gate, "after_attempts": 0, "exchanges": [ex], "debug": [thought]})


def answer(state: DeskState) -> dict:
    reply = state["after_reply"]
    record(
        state,
        "answer",
        by="model",
        memory=F.memory_of(state),
        design=design_now(state),
        read=list(reply.cites) + ([reply.figure] if reply.figure else []),
        note=state.get("message") or "",
    )
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
    record(state, "revise", by="person", memory=memory, design=design_now(state), read=[src], note=", ".join(settled))
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
    before = {a: f.value for a, f in fork.fields.items()}
    rejected = ops.apply(fork, updates, CAT)
    changed = {a: f.value for a, f in fork.fields.items() if before.get(a) != f.value}  # a field the kind change reopened keeps its value: not supposed
    if not changed:
        note = "I could not suppose that: " + "; ".join(rejected)
        return Command(goto="talk", update={"after_reply": reply.model_copy(update={"text": note, "kind": "answer"})})
    record(state, "what_if", by="person", memory=memory, design=design_now(state), read=[src], note=", ".join(f"{a} = {v}" for a, v in changed.items()))
    return Command(
        goto="fit", update={"fork": fork, "what_if": {a: str(v) for a, v in changed.items()}, "handoff": None, "gate_errors": [], "decide_attempts": 0}
    )


def requestion(state: DeskState) -> dict:
    q = state["after_reply"].question or ""
    record(state, "requestion", by="person", memory=F.memory_of(state), design=design_now(state), read=[f"user:turn:{int(state.get('turn') or 0)}"], note=q)
    return {"question": q, "message": q, "phase": "before", "handoff": None, "invalid": None, "prefilter_votes": [], "oriented": False, "focus": []}


def draw_after(state: DeskState) -> dict:
    """The person asked for a picture after the run: the drawing tool makes it from the file, told what the pack settled and what
    the run found, and it lands in the design's folder. The caption is the reply; the picture is shown beside it."""
    reply = state["after_reply"]
    runs = state.get("runs") or []
    cur = runs[-1]
    memory = F.memory_of(state)
    turn = int(state.get("turn") or 0)
    ask = (reply.draw or "").strip()
    context, columns = draw_context(memory)
    if cur.design_dir and (Path(cur.design_dir) / "handoff.json").exists():
        context = (
            Handoff.model_validate_json((Path(cur.design_dir) / "handoff.json").read_text()).render_context()
            + "\n\nCOLUMNS\n"
            + context.split("COLUMNS\n", 1)[-1]
        )
    mat = M.render(cur, memory)
    context += "\n\nWHAT THE RUN FOUND\n" + "\n".join(ln for ln in mat.lines if ln.startswith(("[design.", "[estimate:", "[check:", "[refute:", "[placebo:")))
    req = VD.DrawRequest(
        dataset=memory.name,
        moment="post",
        design=int(cur.index),
        memory_version=memory.version,
        ask=ask,
        context=context,
        csv=_csv_path(memory),
        columns=columns,
    )
    artifact, decline, thoughts = VD.draw(req)
    if artifact is None:
        assert decline is not None
        text = f"I could not draw that: {decline.reason}"
        return {"after_reply": reply.model_copy(update={"kind": "answer", "text": text, "cites": ["run.question"]}), "artifact": None, "debug": thoughts}
    record(
        state,
        "explore",
        by="model",
        memory=memory,
        design=design_now(state),
        read=[f"user:turn:{turn}"],
        left=[str(VS.folder(artifact.dataset, artifact.moment, artifact.design, artifact.id))],
        note=ask,
    )
    text = f"{artifact.caption} [{artifact.address}]"
    return {
        "after_reply": reply.model_copy(update={"kind": "answer", "text": text, "cites": [artifact.address]}),
        "artifact": artifact.model_dump(),
        "debug": thoughts,
    }
