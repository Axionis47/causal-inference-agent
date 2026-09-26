"""The interview before the run: check and probe the memory, compose one question per turn, the ready moment, listen, and
infer what a message settles."""

from __future__ import annotations

from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from causal_agent.common.contracts import QuestionFrame
from causal_agent.common.llm import structured
from causal_agent.desk.contracts import Ask, DeskAnswer, Finding, Inference
from causal_agent.desk.nodes import decide as D
from causal_agent.desk.nodes import frame as F
from causal_agent.desk.nodes.shared import (
    CAT,
    INFER_ATTEMPTS,
    QUIT_WORDS,
    RUN_WORDS,
    TH,
    _columns_in_play,
    _csv_path,
    _design_line,
    _remember,
    _writer,
    focused_needs,
    kinds_text,
)
from causal_agent.desk.prompts import journey as P
from causal_agent.desk.state import Context, DeskState
from causal_agent.families import registry as R
from causal_agent.families.base import Family
from causal_agent.memory import ops, store
from causal_agent.memory import views as V
from causal_agent.memory.catalogue import ClaimKind
from causal_agent.memory.records import COLUMN_KIND, Memory
from causal_agent.profile import data as PD
from causal_agent.viz.spec import Point

# ------------------------------------------------------------------ check, probe, fit (facts)


def check(state: DeskState) -> dict:
    memory = F.memory_of(state)
    df = V.table_of(memory)
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
    df = V.table_of(memory)
    fr = state.get("frame")
    probes = ops.probe(memory, df, R.REGISTRY.values(), TH, CAT)
    needs = focused_needs(state)
    status = ops.fit(memory, probes, needs, columns=_columns_in_play(memory, fr), cat=CAT)
    opened = ops.open(memory, status, needs, CAT)
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
        text = (
            "I read these from what was written about the file. Are they right?\n" + "\n".join(f"• {ln}" for ln in lines) + "\nSay yes, or correct any of them."
        )
        return Ask(
            addresses=[o.address for o in drafts],
            kind="confirm",
            text=text,
            options=["Yes, all right", "No"],
            because=sorted({b for o in drafts for b in o.because}),
        )
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
            text = f"For each of these columns: what does it record, and was it fixed before {ch}, set at it, or measured after it? " + ", ".join(names) + "."
        return Ask(
            addresses=[o.address for o in cols],
            kind="columns",
            text=text,
            options=["before", "at", "after", "unknown"],
            because=sorted({b for o in cols for b in o.because}),
            evidence=[by_addr[o.address].evidence for o in cols if o.address in by_addr],
        )
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


def compose_map(status, memory: Memory, frame: QuestionFrame | None) -> str:
    """What the file could answer, said once after the question is read: each family in play with what it answers and what is
    still to settle for it, each family struck already with why, and that the person may narrow the interview to the ones
    they care about. By code, from the fit grid and the families' knowledge."""
    registry = {f.name: f for f in R.knowledge()}
    q = f"{frame.outcome} against {frame.cause}" if frame and frame.outcome and frame.cause else "the question"
    lines = []
    if status.surviving:
        lines.append(f"With this file, {q} could be answered {len(status.surviving)} way{'s' if len(status.surviving) != 1 else ''}:")
        for fam in status.surviving:
            know = registry.get(fam)
            answers = (know.answers[0].upper() + know.answers[1:]) if know else "a family the registry does not describe"
            todo = [("the columns" if kind == "measured" else kind.replace("_", " ")) for kind, cell in status.table.get(fam, {}).items() if cell == "unknown"]
            lines.append(f"- {fam.replace('_', ' ')}: {answers}." + (f" Still to settle: {', '.join(todo)}." if todo else " Nothing left to settle."))
    else:
        lines.append(f"With this file, no family stands yet for {q}.")
    if status.struck:
        lines.append("Struck already: " + "; ".join(f"{fam.replace('_', ' ')} ({why})" for fam, why in status.struck.items()) + ".")
    lines.append("Say which of these you care about and I will only ask what they need; or answer as we go and every one stays in play.")
    return "\n".join(lines)


def acknowledge(memory: Memory, addresses: list[str]) -> str:
    lines = []
    for a in addresses:
        f = memory.field(a)
        if f is None or f.value is None and f.status != "unknown":
            continue
        lines.append(
            f"[{a}] "
            + (
                "unknown"
                if f.status == "unknown"
                else _value_words(CAT.kinds[Memory.parse(a)[1] if a.startswith("claim:") else COLUMN_KIND], Memory.parse(a)[2], f.value)
            )
        )
    return ("Noted: " + "; ".join(lines) + "\n\n") if lines else ""


def ask(state: DeskState) -> Command[Literal["listen", "fit", "convince"]]:
    st = state["status"]
    if state.get("run_requested") and st.ready:
        return Command(goto="fit", update={"run_requested": False, "ask": None})
    memory = F.memory_of(state)
    a = compose_ask(memory, state.get("open") or [], state.get("findings") or [], state.get("frame"))
    head = (state.get("note") or "") + acknowledge(memory, state.get("settled_now") or [])
    if not state.get("oriented"):  # the first reply after the question is read: the map of what the file could answer
        head = compose_map(st, memory, state.get("frame")) + "\n\n" + head
    if state.get("explained"):  # the person asked the desk something last turn: the answer comes first, then what is asked next
        head = state["explained"] + "\n\n" + head
    if a is None:
        if st.ready:
            return Command(
                goto="convince", update={"ask": None, "reply": head, "run_requested": False, "figure": None, "oriented": True, "explained": None, "note": ""}
            )
        body = (
            "Nothing more to ask, but no design fits yet: "
            + "; ".join(f"{f} ({w})" for f, w in st.struck.items())
            + ". Tell me what is different about the data, or ask another question."
        )
    else:
        body = a.text
        if state.get("run_requested"):
            body = "Before I can run, this still has to be settled. " + body
    return Command(
        goto="listen", update={"ask": a, "reply": head + body, "run_requested": False, "figure": None, "oriented": True, "explained": None, "note": ""}
    )


# ------------------------------------------------------------------ convince (the ready moment)


def _belief_words(memory: Memory, family: Family) -> str:
    """The assumption the family bets on, and the person's own words where a belief carries them."""
    said = []
    for kind in ("unobserved", "exclusion", "spillover", "trend_continues", "cutoff_only"):
        if kind in (R.needs()[family.name].requires if family.name in R.needs() else []):
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
    registry = {f.name: f for f in R.knowledge()}
    head = state.get("reply") or ""
    verdicts = {v.family: v for v in update.get("family_verdicts") or []}
    if d is None or g.goto == "__end__" or d.chosen not in registry:
        text = (
            head
            + "Everything the analysis needs is settled, but no family stands: "
            + "; ".join(f"{v.family} ({next((n.note for n in v.needs if not n.met), v.concern or 'does not fit')})" for v in verdicts.values())
            + ". Tell me what is different about the data."
        )
        return {**update, "reply": text, "figure": None, "handoff": None}
    fam = registry[d.chosen]
    probes = [p for p in update.get("probes") or [] if p.family == fam.name and p.passed is not None]
    evidence = "; ".join(f"{p.detail} [{p.address}]" for p in probes) or "no probe applies"
    fields = [f"[claim:assignment.kind] {memory.value('claim:assignment.kind')}"] + [
        f"[{a}]" for a in ("claim:change.what", "claim:grain.panel") if memory.value(a) is not None
    ]
    struck = [
        f"{v.family}: {next((n.note for n in v.needs if not n.met), v.concern or 'does not fit')}"
        for v in verdicts.values()
        if not v.admissible and v.family != fam.name
    ]
    figure = None
    fig_line = ""
    try:
        from causal_agent.viz.graph import make

        fig = make(
            Point(family=fam.name, claim=fam.convince or fam.answers, about=[p.address for p in probes]),
            memory.name,
            outcome=fr.outcome if fr else None,
            treatment=fr.cause if fr else None,
        )
        if fig.made and fig.spec is not None:
            figure = fig.spec.model_dump()
            fig_line = f"The figure shows it: {fig.spec.note} [{fig.spec.address}]"
        else:
            fig_line = f"No figure could make the point: {fig.why}"
    except Exception as e:  # a figure is never a reason to stop
        fig_line = f"No figure could be made ({type(e).__name__})."
    lines = [
        head + "Everything the analysis needs is settled.",
        f"Design: {fam.name.replace('_', ' ')}. {fam.answers[0].upper() + fam.answers[1:]}.",
        f"It rests on: {_belief_words(memory, fam)}",
        f"Evidence: {evidence}. " + " ".join(fields),
        fig_line,
    ]
    if struck:
        lines.append("Set aside: " + "; ".join(struck) + ".")
    lines.append("Say run to hand off, or tell me anything to change.")
    _writer()({"convince": {"family": fam.name, "figure": bool(figure)}})
    return {**update, "reply": "\n".join(ln for ln in lines if ln), "figure": figure}


def listen(state: DeskState) -> Command[Literal["infer", "fit", "handoff", "check", "__end__"]]:
    st, a = state["status"], state.get("ask")
    payload = {
        "phase": "before",
        "kind": "ask",
        "text": state.get("reply") or "",
        "status": st.render(list(CAT.kinds)) if st else "",
        "ready": bool(st and st.ready),
        "open": list(st.open) if st else [],
        "ask": a.model_dump() if a else None,
        "figure": state.get("figure"),
    }
    answer = str(interrupt(payload) or "").strip()
    low = answer.lower()
    if low in QUIT_WORDS:
        return Command(goto="__end__")
    memory = F.memory_of(state)
    turn = int(state.get("turn") or 0) + 1
    _remember(memory, turn, ", ".join((("lane:" + x) if a.from_lane else x) for x in a.addresses) if a else "", answer)
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


def infer(state: DeskState) -> Command[Literal["infer", "check", "explain"]]:
    memory = F.memory_of(state)
    turn = int(state.get("turn") or 0)
    a = state.get("ask")
    asked = (a.text + "\n(settles: " + ", ".join(a.addresses) + ")") if a else "(no question was asked; the person spoke freely)"
    errs = state.get("infer_errors") or []
    errors = ("\nPREVIOUS UPDATES WERE REJECTED:\n" + "\n".join(f"- {e}" for e in errs) + "\nFix them and return the set again.\n") if errs else ""
    cols = "\n".join(V.brief_of(memory, c).line() for c in memory.columns.values() if not c.facts.constant)
    user = P.INFER_USER.format(
        kinds=kinds_text(),
        memory=memory.render() or "(nothing known yet)",
        asked=asked,
        open=_open_lines(memory, state.get("open") or [], a),
        columns=cols,
        turn=turn,
        message=state.get("message") or "",
        errors=errors,
    )
    out, thought = structured(Inference, P.INFER_SYSTEM, user, node=f"infer:{turn}")
    src = f"user:turn:{turn}"
    updates = [
        ops.Update(address=u.address, value=u.value, status="confirmed", source=src, said=u.said or (state.get("message") or "")[:200], reason=u.reason)
        for u in out.updates
    ]
    for addr in out.confirms:
        f = memory.field(addr)
        if f is not None and f.value is not None and f.status in {"drafted", "refuted"}:
            updates.append(
                ops.Update(address=addr, value=f.value, status="confirmed", source=src, said=(state.get("message") or "")[:200], reason="confirmed as drafted")
            )
    for addr in out.unknown:
        updates.append(ops.Update(address=addr, status="unknown", source=src, said=(state.get("message") or "")[:200]))
    before = {a: (f.value, f.status) for a, f in memory.fields.items()}
    rejected = ops.apply(memory, updates, CAT)
    settled = [a for a, f in memory.fields.items() if before.get(a) != (f.value, f.status)]
    focus_update: dict = {}
    if out.focus is not None:  # the person named the families they care about: known names narrow the interview, unknown ones are refused
        known = R.needs()
        bad = [n for n in out.focus if n not in known]
        if bad:
            rejected.append(f"focus: {', '.join(bad)} is not a family the fit grid knows; one of: {', '.join(known)}")
        else:
            focus_update = {"focus": list(dict.fromkeys(out.focus))}
    attempts = int(state.get("infer_attempts") or 0) + 1
    asked_desk = (out.question or "").strip() or state.get("desk_question") or None  # kept across retries
    _writer()({"infer": {"settled": settled, "rejected": rejected, "attempt": attempts, **({"focus": focus_update["focus"]} if focus_update else {})}})
    store.save(memory)
    if rejected and attempts < INFER_ATTEMPTS:
        return Command(
            goto="infer",
            update={
                "infer_errors": rejected,
                "infer_attempts": attempts,
                "settled_now": settled,
                "debug": [thought],
                "desk_question": asked_desk,
                **focus_update,
            },
        )
    update = {"infer_errors": rejected, "infer_attempts": 0, "settled_now": settled, "debug": [thought], "desk_question": asked_desk, **focus_update}
    if asked_desk:
        return Command(goto="explain", update={**update, "explain_errors": [], "explain_attempts": 0})
    return Command(goto="check", update=update)


# ------------------------------------------------------------------ explain (judgement), gated by the cites


MAX_EXPLAIN_ATTEMPTS = 3


def _plain_cite(c: str) -> str:
    """A cite as the gate reads it: without brackets, and a family named as family:<name> by its name alone."""
    c = c.strip().strip("[]").strip()
    return c[len("family:") :] if c.startswith("family:") else c


def explain(state: DeskState) -> Command[Literal["explain", "check"]]:
    """The person asked the desk something: one answer from the families' knowledge, the fit grid, and the memory, every cite
    checked by code; three tries, then the honest fallback. Shown before the next thing asked."""
    memory = F.memory_of(state)
    st = state.get("status")
    a = state.get("ask")
    question = state.get("desk_question") or ""
    errs = state.get("explain_errors") or []
    errors = ("\nTHE LAST ANSWER WAS REFUSED:\n" + "\n".join(f"- {e}" for e in errs) + "\nAnswer again.\n") if errs else ""
    families = "\n\n".join(f.render() for f in R.knowledge())
    user = P.EXPLAIN_USER.format(
        families=families,
        status=st.render(list(CAT.kinds)) if st else "(not fitted yet)",
        kinds=kinds_text(),
        memory=memory.render() or "(nothing known yet)",
        asked=a.text if a else "(nothing yet)",
        question=question,
        errors=errors,
    )
    out, thought = structured(DeskAnswer, P.EXPLAIN_SYSTEM, user, node="explain")
    names = {f.name for f in R.knowledge()}
    probes = state.get("probes") or []
    cites = [_plain_cite(c) for c in out.cites]
    bad = [c for c in cites if c not in names and not D.resolves(c, memory, probes)]
    problems = []
    if bad:
        problems.append(f"cites that are neither a family name nor an address in the memory: {bad}")
    if not out.cites:
        problems.append("an answer cites at least one family name or memory address")
    attempts = int(state.get("explain_attempts") or 0) + 1
    _writer()({"explain": {"question": question, "attempt": attempts, "problems": problems}})
    if problems and attempts < MAX_EXPLAIN_ATTEMPTS:
        return Command(goto="explain", update={"explain_errors": problems, "explain_attempts": attempts, "debug": [thought]})
    if problems:
        text = "I can only answer that from what is settled. " + (_design_line(st, memory, state.get("frame")) if st else "")
    else:
        text = out.text.strip() + (" " + " ".join(f"[{c}]" for c in dict.fromkeys(cites) if c not in out.text) if cites else "")
    return Command(goto="check", update={"explained": text.strip(), "desk_question": None, "explain_errors": [], "explain_attempts": 0, "debug": [thought]})
