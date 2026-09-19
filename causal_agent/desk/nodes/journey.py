"""The journey to ready. Facts: load, read_question's validation, check, probe_fit, ask, the writes. One judgement each,
gated: read_question's frame, infer. Interrupts: ask_question, listen.

The first thing asked is the causal question, and it is validated against the file before anything else. Then one
question per turn, from what the surviving families still need; every answer goes through `infer` and the gate into the
memory; the file's checks refute what they can, and a refuted answer is asked again with the number that refuted it."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from causal_agent.common.contracts import QuestionFrame, Said
from causal_agent.common.llm import structured
from causal_agent.desk import handoff as H
from causal_agent.desk import pipeline
from causal_agent.desk.contracts import Ask, Finding, Inference, RunRecord
from causal_agent.desk.nodes import decide as D
from causal_agent.desk.nodes import frame as F
from causal_agent.desk.prompts import journey as P
from causal_agent.desk.state import Context, DeskState
from causal_agent.memory import ops, store
from causal_agent.memory.catalogue import Catalogue, ClaimKind, load_catalogue, load_thresholds
from causal_agent.memory.records import COLUMN_KIND, Memory
from causal_agent.knowledge import Family, load_registry
from causal_agent.profile import data as PD
from causal_agent.profile import datasets as DS
from causal_agent.viz.spec import Point

CAT: Catalogue = load_catalogue()
TH: dict = load_thresholds()
INFER_ATTEMPTS = 3
QUIT_WORDS = {"quit", "exit"}
RUN_WORDS = {"run", "go", "run it", "run the analysis"}


def _writer():
    try:
        return get_stream_writer()
    except Exception:  # outside a graph run
        return lambda *_: None


# ------------------------------------------------------------------ helpers


def kinds_text() -> str:
    """The catalogue as the judgements read it: every kind, its fields, the legal values, the hints."""
    out = []
    for k in CAT.ordered():
        fields = "; ".join(
            f"{name} ({spec.type}{': ' + ', '.join(f'{o} = {spec.about[str(o)]}' if str(o) in spec.about else str(o) for o in spec.options) if spec.options else ''}{', optional' if spec.optional else ''})"
            + (f" = {spec.hint}" if spec.hint else "")
            for name, spec in k.fields.items()
        )
        tag = " [per column]" if k.per_column else " [uncheckable: only on the person's word]" if k.uncheckable else ""
        cues = f" Reading their words: {k.cues}" if k.cues else ""
        out.append(f"- {k.name}{tag}: {k.about}. Fields: {fields}.{cues}")
    return "\n".join(out)


def _csv_path(memory: Memory) -> Path:
    csv = memory.csv or (DS.dataset_entries().get(memory.name) or {}).get("csv")
    return Path(csv) if csv and Path(csv).is_absolute() else Path(DS.ROOT) / csv


def _entry(memory: Memory) -> dict:
    return DS.dataset_entries().get(memory.name) or {}


def _columns_in_play(memory: Memory, frame: QuestionFrame | None) -> list[str]:
    return H.in_play(memory, frame, _entry(memory))


def _remember(memory: Memory, turn: int, about: str, text: str) -> None:
    if text and not any(s.turn == turn for s in memory.said):
        memory.said.append(Said(turn=turn, about=about, text=text))


def _design_line(status, memory: Memory, frame: QuestionFrame | None) -> str:
    fams = status.surviving
    if not fams:
        return ""
    ch = memory.values_of("claim:change")
    return f"The question, {frame.outcome if frame else 'the outcome'} against {ch.get('what') or 'the change'}, can be answered by {', '.join(f.replace('_', ' ') for f in fams)}."


# ------------------------------------------------------------------ load and the question


def load(state: DeskState) -> dict:
    F.memory_of(state)  # raises for an unknown dataset
    return {"turn": int(state.get("turn") or 0), "phase": "before", "invalid": None, "frame_attempts": 0, "findings": [], "refutations": {}, "settled_now": [],
            "ask": None, "open": [], "status": None, "infer_errors": [], "infer_attempts": 0, "run_requested": False, "ready": False,
            "runs": list(state.get("runs") or []), "prefilter_votes": [], "family_verdicts": [], "probes": [], "fit_status": None, "gate_errors": [],
            "decide_attempts": 0, "debug": [], "after_reply": None, "after_errors": [], "after_attempts": 0, "brief": ""}


def ask_question(state: DeskState) -> Command[Literal["mine", "__end__"]]:
    """The first turn, and every turn after a question that did not pass: the causal question, nothing before it."""
    memory = F.memory_of(state)
    invalid = state.get("invalid")
    if invalid:
        text = invalid
    else:
        cols = ", ".join(c.name for c in memory.columns.values())
        rows = memory.facts.get("rows")
        text = (f"What is the causal question you want answered from this file? Say what changed and what it might have affected. "
                f"The file has {rows} rows and these columns: {cols}.")
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
        out.append({"driver_search": "it asks what drives an outcome rather than what one change did to it",
                    "root_cause": "it asks why something happened rather than what one change did to an outcome",
                    "not_causal": "it does not ask what a change did to an outcome"}[frame.intent])
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
        text = "That is not yet a question I can answer from this file: " + "; ".join(problems) + ". Ask it again, naming the change and the outcome as they appear in the columns."
        return Command(goto="ask_question", update={"frame": fr, "invalid": text, "frame_attempts": attempts, "debug": out["debug"]})
    note = ""
    old = memory.value("claim:assignment.treatment_column")
    if state.get("runs") and fr.cause and old and memory.column(old) is not None and memory.column(fr.cause) is not None and memory.column(old).key != memory.column(fr.cause).key:
        # a new change: everything settled relative to the old one is asked again; what a column is carries over
        dropped = ops.forget_change(memory)
        note = f"The change is not the one before ({old}), so what was settled relative to it is asked again ({len(dropped)} fields); what each column is carries over. "
    if fr.cause and memory.value("claim:assignment.treatment_column") is None:  # the frame's reading, as a draft the person confirms
        ops.apply(memory, [ops.Update(address="claim:assignment.treatment_column", value=fr.cause, status="drafted", source="code:frame")], CAT)
    store.save(memory)
    return Command(goto="check", update={"frame": fr, "invalid": None, "frame_attempts": attempts, "debug": out["debug"], "settled_now": [], "reply": note})


# ------------------------------------------------------------------ check, probe, fit (facts)


def check(state: DeskState) -> dict:
    memory = F.memory_of(state)
    df = F.table_of(memory)
    prof = PD.profile_for(_csv_path(memory))
    fr = state.get("frame")
    turn = int(state.get("turn") or 0)
    findings = ops.check(memory, df, prof, TH, CAT, outcome=fr.outcome if fr else None, treatment=fr.cause if fr else None)
    refs = dict(state.get("refutations") or {})
    max_asks = int(TH["interview"]["max_asks"])
    for fd in findings:
        if fd.passed is False:
            f = memory.field(fd.address)
            if f is not None and f.source == f"user:turn:{turn}":  # the person answered this turn and the file says otherwise
                refs[fd.address] = refs.get(fd.address, 0) + 1
                if refs[fd.address] >= max_asks:
                    f.status = "contradiction"
    store.save(memory)
    return {"findings": [Finding.of(f) for f in findings], "refutations": refs}


def probe_fit(state: DeskState) -> dict:
    memory = F.memory_of(state)
    df = F.table_of(memory)
    fr = state.get("frame")
    probes = ops.probe(memory, df, TH, CAT)
    status = ops.fit(memory, probes, CAT, columns=_columns_in_play(memory, fr))
    opened = ops.open(memory, status, CAT)
    _writer()({"fit": {"surviving": status.surviving, "struck": status.struck, "open": [o.address for o in opened], "ready": status.ready}})
    return {"probes": probes, "status": status, "fit_status": status.model_dump(), "open": opened, "ready": status.ready}


# ------------------------------------------------------------------ ask (code)


def _options_text(kind: ClaimKind, field: str) -> str:
    spec = kind.fields[field]
    if spec.type == "choice":
        return "; ".join(f"{o} ({spec.about[str(o)]})" if str(o) in spec.about else str(o) for o in spec.options)
    if spec.type == "bool":
        return "yes or no"
    return ""


def _value_words(kind: ClaimKind, field: str, value) -> str:
    spec = kind.fields.get(field)
    if spec and spec.type == "choice" and str(value) in spec.about:
        return f"{value} ({spec.about[str(value)]})"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _change_words(memory: Memory) -> str:
    return memory.value("claim:change.what") or "the change"


def compose_ask(memory: Memory, opened: list, findings: list[Finding], frame: QuestionFrame | None) -> Ask | None:
    """One question from what is open: the drafts to confirm first, then the columns in one tick, then one dataset field."""
    if not opened:
        return None
    by_addr = {f.address: f for f in findings if f.passed is False}
    drafts = [o for o in opened if o.status == "drafted"]
    if drafts:
        lines = []
        for o in drafts:
            kind = CAT.kinds[o.kind]
            f = memory.field(o.address)
            if o.address.startswith("col:"):
                col = memory.column(o.address[4:].split(".")[0])
                lines.append(f"{col.name if col else o.address}: {o.field} = {_value_words(kind, o.field, f.value)}")
            else:
                lines.append(f"{o.kind}, {o.field}: {_value_words(kind, o.field, f.value)}")
        text = "I read these from what was written about the file. Are they right?\n" + "\n".join(f"• {ln}" for ln in lines) + "\nSay yes, or correct any of them."
        return Ask(addresses=[o.address for o in drafts], kind="confirm", text=text, options=["Yes, all right", "No"], because=sorted({b for o in drafts for b in o.because}))
    cols = [o for o in opened if o.address.startswith("col:") and o.field in ("meaning", "when")]
    if cols:
        names, refuted = [], []
        for o in cols:
            col = memory.column(o.address[4:].split(".")[0])
            name = col.name if col else o.address
            if name not in names:
                names.append(name)
            fd = by_addr.get(o.address)
            if fd is not None:
                refuted.append(f"{name}: you said {memory.value(o.address)}, but {fd.detail} [{fd.evidence}]")
        ch = _change_words(memory)
        if refuted:
            text = "The file disagrees with what you said about " + "; ".join(refuted) + ". Which is it?"
        else:
            text = (f"For each of these columns: what does it record, and was it fixed before {ch}, set at it, or measured after it? "
                    + ", ".join(names) + ".")
        return Ask(addresses=[o.address for o in cols], kind="columns", text=text, options=["before", "at", "after", "unknown"],
                   because=sorted({b for o in cols for b in o.because}), evidence=[by_addr[o.address].evidence for o in cols if o.address in by_addr])
    o = opened[0]
    kind = CAT.kinds[o.kind]
    fd = by_addr.get(o.address)
    because = list(o.because)
    if o.address.startswith("col:"):
        col = memory.column(o.address[4:].split(".")[0])
        name = col.name if col else o.address
        hint = kind.fields[o.field].hint or o.field
        text = f"About '{name}': {hint}?"
    elif fd is not None:
        text = f"You said {o.kind}, {o.field} = {_value_words(kind, o.field, memory.value(o.address))}, but {fd.detail} [{fd.evidence}]. Which is it?"
    else:
        hint = kind.fields[o.field].hint
        required = kind.required(memory.values_of(o.address.rsplit(".", 1)[0]))
        text = kind.frame[0].upper() + kind.frame[1:] + "?"
        if len(required) > 1 or o.field not in required:  # the frame covers several fields: say which one this turn settles
            text += f" This turn: {o.field.replace('_', ' ')}" + (f", {hint}" if hint else "") + "."
        elif hint:
            text += f" ({hint})"
    opts = _options_text(kind, o.field)
    if opts:
        text += f" Answer {opts}." if opts == "yes or no" else f" One of: {opts}."
    kind_word: str = "confirm" if fd is not None else ("choose" if o.options else "open")
    return Ask(addresses=[o.address], kind=kind_word, text=text, options=list(o.options), because=because, evidence=[fd.evidence] if fd else [])


def acknowledge(memory: Memory, addresses: list[str]) -> str:
    lines = []
    for a in addresses:
        f = memory.field(a)
        if f is None or f.value is None and f.status != "unknown":
            continue
        lines.append(f"[{a}] " + ("unknown" if f.status == "unknown" else _value_words(CAT.kinds[Memory.parse(a)[1] if a.startswith('claim:') else COLUMN_KIND], Memory.parse(a)[2], f.value)))
    return ("Noted: " + "; ".join(lines) + "\n\n") if lines else ""


def ask(state: DeskState) -> Command[Literal["listen", "fit", "convince"]]:
    st = state["status"]
    if state.get("run_requested") and st.ready:
        return Command(goto="fit", update={"run_requested": False, "ask": None})
    memory = F.memory_of(state)
    a = compose_ask(memory, state.get("open") or [], state.get("findings") or [], state.get("frame"))
    head = (state.get("reply") or "" if not state.get("ask") and not state.get("settled_now") else "") + acknowledge(memory, state.get("settled_now") or [])
    if a is None:
        if st.ready:
            return Command(goto="convince", update={"ask": None, "reply": head, "run_requested": False, "figure": None})
        body = "Nothing more to ask, but no design fits yet: " + "; ".join(f"{f} ({w})" for f, w in st.struck.items()) + ". Tell me what is different about the data, or ask another question."
    else:
        body = a.text
        if state.get("run_requested"):
            body = "Before I can run, this still has to be settled. " + body
    return Command(goto="listen", update={"ask": a, "reply": head + body, "run_requested": False, "figure": None})


# ------------------------------------------------------------------ convince (the ready moment)


def _belief_words(memory: Memory, family: Family) -> str:
    """The assumption the family bets on, and the person's own words where a belief carries them."""
    said = []
    for kind in ("unobserved", "exclusion", "spillover", "trend_continues", "cutoff_only"):
        if kind in (CAT.families.get(family.name).requires if CAT.families.get(family.name) else []):
            for n, f in memory.fields_of(f"claim:{kind}").items():
                if f.said and f.value is not None:
                    said.append(f'"{f.said}" [claim:{kind}.{n}]')
                    break
    return family.assumes + (" You said: " + "; ".join(said) + "." if said else "")


def convince(state: DeskState, runtime: Runtime[Context]) -> dict:
    """At ready: decide by code (a judgement only among several), make the family's point visible, and say the design in the
    question's words with the evidence, the assumption, the figure, and the struck families with one reason each."""
    memory = F.memory_of(state)
    fr = state.get("frame")
    out = D.fit(state, runtime)
    st2 = {**state, **out}
    dec = D.decide(st2, runtime)
    st3 = {**st2, **dec}
    g = D.gate(st3, runtime)
    update: dict = {**out, **dec, **(g.update or {}), "convinced_version": memory.version}
    d = update.get("decision")
    registry = {f.name: f for f in load_registry(runtime.context.registry_path if runtime and runtime.context else None)}
    head = state.get("reply") or ""
    verdicts = {v.family: v for v in update.get("family_verdicts") or []}
    if d is None or g.goto == "__end__" or d.chosen not in registry:
        text = head + "Everything the analysis needs is settled, but no family stands: " + "; ".join(
            f"{v.family} ({next((n.note for n in v.needs if not n.met), v.concern or 'does not fit')})" for v in verdicts.values()) + ". Tell me what is different about the data."
        return {**update, "reply": text, "figure": None, "handoff": None}
    fam = registry[d.chosen]
    probes = [p for p in update.get("probes") or [] if p.family == fam.name and p.passed is not None]
    evidence = "; ".join(f"{p.detail} [{p.address}]" for p in probes) or "no probe applies"
    fields = [f"[claim:assignment.kind] {memory.value('claim:assignment.kind')}"] + [f"[{a}]" for a in ("claim:change.what", "claim:grain.panel") if memory.value(a) is not None]
    struck = [f"{v.family}: {next((n.note for n in v.needs if not n.met), v.concern or 'does not fit')}" for v in verdicts.values() if not v.admissible and v.family != fam.name]
    figure = None
    fig_line = ""
    try:
        from causal_agent.viz.graph import make

        fig = make(Point(family=fam.name, claim=fam.convince or fam.answers, about=[p.address for p in probes]), memory.name, outcome=fr.outcome if fr else None,
                   treatment=fr.cause if fr else None)
        if fig.made and fig.spec is not None:
            figure = fig.spec.model_dump()
            fig_line = f"The figure shows it: {fig.spec.note} [{fig.spec.address}]"
        else:
            fig_line = f"No figure could make the point: {fig.why}"
    except Exception as e:  # a figure is never a reason to stop
        fig_line = f"No figure could be made ({type(e).__name__})."
    lines = [head + "Everything the analysis needs is settled.",
             f"Design: {fam.name.replace('_', ' ')}. {fam.answers[0].upper() + fam.answers[1:]}.",
             f"It rests on: {_belief_words(memory, fam)}",
             f"Evidence: {evidence}. " + " ".join(fields),
             fig_line]
    if struck:
        lines.append("Set aside: " + "; ".join(struck) + ".")
    lines.append("Say run to hand off, or tell me anything to change.")
    _writer()({"convince": {"family": fam.name, "figure": bool(figure)}})
    return {**update, "reply": "\n".join(ln for ln in lines if ln), "figure": figure}


def listen(state: DeskState) -> Command[Literal["infer", "fit", "handoff", "check", "__end__"]]:
    st, a = state["status"], state.get("ask")
    payload = {"phase": "before", "kind": "ask", "text": state.get("reply") or "", "status": st.render(list(CAT.kinds)) if st else "", "ready": bool(st and st.ready),
               "open": list(st.open) if st else [], "ask": a.model_dump() if a else None, "figure": state.get("figure")}
    answer = str(interrupt(payload) or "").strip()
    low = answer.lower()
    if low in QUIT_WORDS:
        return Command(goto="__end__")
    memory = F.memory_of(state)
    turn = int(state.get("turn") or 0) + 1
    _remember(memory, turn, ", ".join(a.addresses) if a else "", answer)
    if low in RUN_WORDS:
        if st and st.ready:
            store.save(memory)
            if state.get("decision") is not None and state.get("convinced_version") == memory.version:  # decided at the ready moment; nothing moved since
                return Command(goto="handoff", update={"turn": turn, "message": answer, "run_requested": False})
            return Command(goto="fit", update={"turn": turn, "message": answer, "run_requested": False})
        # "run" is the person's word that the drafts they were shown stand; empty and refuted fields stay open
        confirmed = []
        for o in state.get("open") or []:
            f = memory.field(o.address)
            if o.status == "drafted" and f is not None and f.value is not None:
                memory.set(o.address, f.value, status="confirmed", source=f"user:turn:{turn}", said=answer)
                confirmed.append(o.address)
        store.save(memory)
        return Command(goto="check", update={"turn": turn, "message": answer, "run_requested": True, "settled_now": confirmed})
    store.save(memory)
    return Command(goto="infer", update={"turn": turn, "message": answer, "infer_errors": [], "infer_attempts": 0, "settled_now": []})


# ------------------------------------------------------------------ infer (judgement), gated by apply


def _open_lines(memory: Memory, opened: list, asked: Ask | None) -> str:
    lines = []
    order = (asked.addresses if asked else []) + [o.address for o in opened if not asked or o.address not in asked.addresses]
    by = {o.address: o for o in opened}
    for a in order:
        o = by.get(a)
        if o is None:
            kind_name = Memory.parse(a)[1] if a.startswith("claim:") else COLUMN_KIND
            field = Memory.parse(a)[2]
            kind = CAT.kinds.get(kind_name)
            spec = kind.fields.get(field) if kind and field else None
            lines.append(f"{a} · {spec.hint if spec and spec.hint else (kind.frame if kind else '')} · {_options_text(kind, field) if kind and spec else ''}")
            continue
        kind = CAT.kinds[o.kind]
        spec = kind.fields[o.field]
        lines.append(f"{a} · {spec.hint or kind.frame} · {_options_text(kind, o.field) or spec.type}" + (" (optional)" if o.optional else ""))
    return "\n".join(lines) or "(nothing open)"


def infer(state: DeskState) -> Command[Literal["infer", "check"]]:
    memory = F.memory_of(state)
    turn = int(state.get("turn") or 0)
    a = state.get("ask")
    asked = (a.text + "\n(settles: " + ", ".join(a.addresses) + ")") if a else "(no question was asked; the person spoke freely)"
    errs = state.get("infer_errors") or []
    errors = ("\nPREVIOUS UPDATES WERE REJECTED:\n" + "\n".join(f"- {e}" for e in errs) + "\nFix them and return the set again.\n") if errs else ""
    cols = "\n".join(H.brief_of(memory, c).line() for c in memory.columns.values() if not c.facts.constant)
    user = P.INFER_USER.format(kinds=kinds_text(), memory=memory.render() or "(nothing known yet)", asked=asked, open=_open_lines(memory, state.get("open") or [], a),
                               columns=cols, turn=turn, message=state.get("message") or "", errors=errors)
    out, thought = structured(Inference, P.INFER_SYSTEM, user, node=f"infer:{turn}")
    src = f"user:turn:{turn}"
    updates = [ops.Update(address=u.address, value=u.value, status="confirmed", source=src, said=u.said or (state.get("message") or "")[:200], reason=u.reason) for u in out.updates]
    for addr in out.confirms:
        f = memory.field(addr)
        if f is not None and f.value is not None and f.status in {"drafted", "refuted"}:
            updates.append(ops.Update(address=addr, value=f.value, status="confirmed", source=src, said=(state.get("message") or "")[:200], reason="confirmed as drafted"))
    for addr in out.unknown:
        updates.append(ops.Update(address=addr, status="unknown", source=src, said=(state.get("message") or "")[:200]))
    before = {a: (f.value, f.status) for a, f in memory.fields.items()}
    rejected = ops.apply(memory, updates, CAT)
    settled = [a for a, f in memory.fields.items() if before.get(a) != (f.value, f.status)]
    attempts = int(state.get("infer_attempts") or 0) + 1
    _writer()({"infer": {"settled": settled, "rejected": rejected, "attempt": attempts}})
    store.save(memory)
    if rejected and attempts < INFER_ATTEMPTS:
        return Command(goto="infer", update={"infer_errors": rejected, "infer_attempts": attempts, "settled_now": settled, "debug": [thought]})
    return Command(goto="check", update={"infer_errors": rejected, "infer_attempts": 0, "settled_now": settled, "debug": [thought]})


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
        rec = RunRecord(index=n, dataset=state["dataset"], question=question, status="no_handoff", decision=_decision(state), decision_record=state.get("decision_record") or "",
                        design_dir=state.get("design_dir"))
    else:
        rec = pipeline.run(Path(state["design_dir"]) / "handoff.json", n, state["dataset"], question, decision=_decision(state), decision_record=state.get("decision_record") or "")
    rec.what_if = dict(state.get("what_if") or {})
    return {"runs": runs + [rec], "phase": "after"}
