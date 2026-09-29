"""The interview before the run: check and probe the memory, the story, the readback, one gap question per turn asked because a
decision needs it, the ready moment, listen, the Reader over what was said, the drawing tool, and the Explainer."""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from causal_agent.common.contracts import QuestionFrame
from causal_agent.desk import designer as DG
from causal_agent.desk import explainer as X
from causal_agent.desk import reader as RD
from causal_agent.desk.contracts import Ask, Finding
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
    design_now,
    focused_needs,
    journal_of,
    kind_of,
    options_text,
    record,
    probes_and_facts,
    value_words,
)
from causal_agent.desk.readback import compose_readback
from causal_agent.desk.state import Context, DeskState
from causal_agent.families import registry as R
from causal_agent.families.base import Decision, Family
from causal_agent.memory import ops, store
from causal_agent.memory import views as V
from causal_agent.memory.matrix import Matrix
from causal_agent.memory.records import Memory
from causal_agent.profile import data as PD
from causal_agent.viz import draw as VD

STORY_TEXT = (
    "Tell me the story: what the change was, who could get it and how that was decided, what each column records and when it "
    "was set, and what one row is. Paste a note if you have one."
)

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
    needs = focused_needs(state)
    columns = _columns_in_play(memory, fr)
    status = ops.fit(memory, probes, needs, columns=columns, cat=CAT)
    probes = probes_and_facts(memory, df, fr, columns)
    opened = ops.open(memory, status, needs, CAT, columns=columns, exclude=[x for x in (fr.outcome if fr else None, fr.cause if fr else None) if x])
    prev = state.get("matrix") if isinstance(state.get("matrix"), Matrix) else Matrix()
    matrix = prev.update(memory, probes, needs, columns=columns, cat=CAT)
    changed = matrix.diff(prev)
    if changed:  # the matrix is a record: a cell that moved is a step of the conversation, with what moved it
        record(
            state,
            "fit",
            by="code",
            memory=memory,
            design=design_now(state),
            read=list(dict.fromkeys(c.set_by for c in changed if c.set_by))[:12],
            note="; ".join(c.line() for c in changed),
        )
    _writer()({"fit": {"surviving": status.surviving, "struck": status.struck, "open": [o.address for o in opened], "ready": status.ready}})
    return {"probes": probes, "status": status, "fit_status": status.model_dump(), "matrix": matrix, "open": opened, "ready": status.ready}


# ------------------------------------------------------------------ ask (code)


def _change_words(memory: Memory) -> str:
    return memory.value("claim:change.what") or "the change"


def _pattern(p: str) -> str:
    """A decision's rests_on pattern as fnmatch reads it: <column> and its kin stand for any column."""
    return re.sub(r"<[^>]*>", "*", p)


def rests_on(decision: Decision, address: str) -> bool:
    return any(fnmatchcase(address, _pattern(p)) for p in decision.rests_on)


def decision_for(address: str, families: list[str]) -> tuple[Family, Decision] | None:
    """The first decision, among the families given in order, that rests on this address."""
    registry = {f.name: f for f in R.knowledge()}
    for name in families:
        fam = registry.get(name)
        if fam is None:
            continue
        for d in fam.decisions:
            if rests_on(d, address):
                return fam, d
    return None


def _col_ask(field: str, change: str) -> str:
    """What a per-column field asks, in the world's words."""
def first_gap(gaps: list, families: list[str]) -> tuple:
    """The gap asked next, and the decision it serves: the decisions of the families in play, in the order each family lists them,
    the first that rests on an open field; a field no decision rests on waits until every decision is served. A refuted field
    comes first regardless, so the person hears the check that refuted it."""
    registry = {f.name: f for f in R.knowledge()}
    blocking = [g for g in gaps if not g.optional]
    for pool in (blocking, gaps):  # what blocks readiness first; a relation a decision rests on, which never blocks, after
        for name in families:
            fam = registry.get(name)
            if fam is None:
                continue
            for d in fam.decisions:
                hit = next((g for g in pool if rests_on(d, g.address)), None)
                if hit is not None:
                    return hit, (fam, d)
        if pool:
            return pool[0], None
    return gaps[0], None


    if field == "when":
        return f"was it fixed before {change}, set at it, or measured after it"
    spec = CAT.kinds["measured"].fields[field]
    return spec.hint or field.replace("_", " ")


def _refuted_line(memory: Memory, o, fd: Finding) -> str:
    kind = CAT.kinds[o.kind]
    if o.address.startswith("col:"):
        col = memory.column(o.address[4:].split(".")[0])
        who = f"'{col.name if col else o.address}'"
    else:
        who = f"{o.kind.replace('_', ' ')}, {o.field.replace('_', ' ')}"
    said = value_words(kind, o.field, memory.value(o.address))
    return f"The file disagrees with what you said about {who}: you said {said}, but {fd.detail} [{fd.evidence}]. Which is it?"


def _claim_line(o, lone: bool) -> str:
    """One claim field as a question: its hint, or the kind's frame when it has none; the legal values after."""
    kind = CAT.kinds[o.kind]
    hint = kind.fields[o.field].hint
    head = f"{o.kind.replace('_', ' ')}, {o.field.replace('_', ' ')}"
    body = hint or (kind.frame if lone else "")
    opts = options_text(kind, o.field)
    tail = (" Answer yes or no." if opts == "yes or no" else f" One of: {opts}.") if opts else ""
    return f"{head}: {body}." + tail if body else f"{head}." + tail


def _belief_text(fam: Family | None, beliefs: list, memory: Memory) -> str:
    """A belief asked as what the design would bet on, with the person's own relevant facts beside it."""
    assumes = fam.assumes if fam is not None else ""
    frames = []
    for o in beliefs:
        kind = CAT.kinds[o.kind]
        q = kind.frame[0].upper() + kind.frame[1:] + "?"
        if q not in frames:
            frames.append(q)
    facts = []
    rule, dep = memory.value("claim:assignment.rule"), memory.value("claim:assignment.depends_on")
    if rule:
        facts.append(f"you said the rule was: {rule}")
    if dep:
        facts.append(f"it depended on {', '.join(dep)}")
    return (f"The design will assume {assumes}. " if assumes else "") + " ".join(frames) + (f" ({'; '.join(facts)}.)" if facts else "")


def compose_ask(memory: Memory, opened: list, findings: list[Finding], frame: QuestionFrame | None, surviving: list[str] | None = None) -> Ask | None:
    """One question from what is open: the readback of the drafts first; then the first open field, asked because a decision
    needs it, together with every open field that decision rests on. A field no decision rests on is asked with the rest of
    its claim, or with the same field of every other column. A refuted field is asked first, with the check that refuted it.
    A belief is asked as what the design would bet on."""
    if not opened:
        return None
    readback = compose_readback(memory, opened)
    if readback is not None:
        return readback
    by_addr = {f.address: f for f in findings if f.passed is False}
    gaps = sorted((o for o in opened if o.status != "drafted"), key=lambda o: o.address not in by_addr)
    if not gaps:
        return None
    families = list(surviving or []) or list(gaps[0].because)
    refuted = gaps[0].address in by_addr
    if refuted:
        o = gaps[0]
        found = decision_for(o.address, families)
    else:
        o, found = first_gap(gaps, families)
    fam, dec = found if found else (None, None)
    if refuted:
        group = [x for x in gaps if x.address in by_addr and (dec is None or rests_on(dec, x.address))] or [o]
    elif dec is not None:
        group = [x for x in gaps if rests_on(dec, x.address)]
    elif CAT.kinds[o.kind].per_column:
        group = [x for x in gaps if x.kind == o.kind and x.field == o.field]
    else:
        prefix = o.address.rsplit(".", 1)[0]
        group = [x for x in gaps if x.address.rsplit(".", 1)[0] == prefix]
    if fam is None and o.because:
        fam = next((f for f in R.knowledge() if f.name == o.because[0]), None)
    ch = _change_words(memory)
    beliefs = [x for x in group if CAT.kinds[x.kind].uncheckable]
    plain = [x for x in group if x not in beliefs]
    lines: list[str] = []
    by_field: dict[str, list[str]] = {}
    for x in plain:
        fd = by_addr.get(x.address)
        if fd is not None:
            lines.append(_refuted_line(memory, x, fd))
        elif x.address.startswith("col:"):
            col = memory.column(x.address[4:].split(".")[0])
            by_field.setdefault(x.field, []).append(col.name if col else x.address)
        else:
            lines.append(_claim_line(x, lone=len(group) == 1 or dec is None))
    for field, names in by_field.items():
        lines.append(f"for each of these columns, {_col_ask(field, ch)}: {', '.join(names)}")
    if dec is None and not beliefs and plain and not any(x.address.startswith("col:") for x in plain) and len(plain) > 1:
        kind = CAT.kinds[o.kind]  # one claim, several fields: the frame once, then which fields this turn settles
        lines = [kind.frame[0].upper() + kind.frame[1:] + "? This turn: " + ", ".join(x.field.replace("_", " ") for x in plain) + "."]
    head = f"To settle {dec.asks}, I need: " if dec is not None else ""
    body = " ".join(ln if ln.endswith((".", "?")) else ln + "." for ln in lines)
    if beliefs:
        body = (body + " " if body else "") + _belief_text(fam, beliefs, memory)
    text = head + body
    if not any(x.address in by_addr for x in group):
        text += " Say don't know for anything you cannot say."
    lone = len(group) == 1
    kind_word: Literal["choose", "open"] = "choose" if lone and group[0].options else "open"
    return Ask(
        addresses=[x.address for x in group],
        kind=kind_word,
        text=text,
        options=list(group[0].options) if lone else [],
        because=sorted({b for x in group for b in x.because}),
        decision=dec.name if dec is not None else "",
        evidence=[by_addr[x.address].evidence for x in group if x.address in by_addr],
    )


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
        lines.append(f"[{a}] " + ("unknown" if f.status == "unknown" else value_words(kind_of(a), Memory.parse(a)[2], f.value)))
    return ("Noted: " + "; ".join(lines) + "\n\n") if lines else ""


def _told_by_note(memory: Memory, opened: list) -> bool:
    """Whether a note already told the story: a draft among the open fields comes from doc:<name>."""
    for o in opened:
        f = memory.field(o.address)
        if o.status == "drafted" and f is not None and (f.source or "").startswith("doc:"):
            return True
    return False


def ask(state: DeskState) -> Command[Literal["listen", "fit", "convince"]]:
    """The next thing asked: once per question the story (unless a note told it), then the readback of the drafts, then one gap
    question per turn asked because a decision needs it. The map comes first, once; whatever the desk answered or drew last
    turn comes before the question."""
    st = state["status"]
    if state.get("run_requested") and st.ready:
        return Command(goto="fit", update={"run_requested": False, "ask": None})
    memory = F.memory_of(state)
    opened = state.get("open") or []
    flags: dict = {}
    a: Ask | None = None
    if not state.get("story_asked") and opened:
        flags["story_asked"] = True
        if not _told_by_note(memory, opened):
            a = Ask(addresses=[o.address for o in opened], kind="story", text=STORY_TEXT, because=sorted({b for o in opened for b in o.because}))
    if a is None:
        a = compose_ask(memory, opened, state.get("findings") or [], state.get("frame"), surviving=list(st.surviving))
        if a is not None and a.kind == "confirm":
            flags["readback_done"] = True
    head = (state.get("note") or "") + acknowledge(memory, state.get("settled_now") or [])
    if not state.get("oriented"):  # the first reply after the question is read: the map of what the file could answer
        head = compose_map(st, memory, state.get("frame")) + "\n\n" + head
    if state.get("explained"):  # the person asked the desk something last turn: the answer comes first, then what is asked next
        head = state["explained"] + "\n\n" + head
    if state.get("drawn"):  # the person asked for a picture last turn: the caption comes first, the picture beside the reply
        head = state["drawn"] + "\n\n" + head
    shown = {"artifact": state.get("artifact"), "drawn": None, **flags}
    if a is None:
        if st.ready:
            return Command(
                goto="convince", update={"ask": None, "reply": head, "run_requested": False, "oriented": True, "explained": None, "note": "", **shown}
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
    return Command(goto="listen", update={"ask": a, "reply": head + body, "run_requested": False, "oriented": True, "explained": None, "note": "", **shown})


# ------------------------------------------------------------------ convince (the ready moment)


def convince(state: DeskState, runtime: Runtime[Context]) -> dict:
    """At ready: decide by code (a judgement only among several), the Designer's brief, and the design said in the question's
    words with the evidence, what it bets on, each decision, the threats, and the struck families with one reason each."""
    memory = F.memory_of(state)
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
        return {**update, "reply": text, "handoff": None, "brief": None}
    fam = registry[d.chosen]
    designed = DG.design_brief({**st3, **(g.update or {})}, runtime)
    brief = designed["brief"]
    update.update({"brief": brief, "debug": list(dec.get("debug") or []) + list(designed.get("debug") or [])})
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
    lines = [
        head + "Everything the analysis needs is settled.",
        f"Design: {fam.name.replace('_', ' ')}. {fam.answers[0].upper() + fam.answers[1:]}.",
        f"It bets on: {brief.bets_on if brief else fam.assumes}",
    ]
    if brief is not None:
        lines += [f"[{dm.address}] {brief.road + ': ' if dm.name == 'road' and brief.road else ''}{dm.choice}" for dm in brief.decisions]
        if brief.road is not None and brief.decision("road") is None:
            lines.append(f"[design.brief.road] {brief.road}")
        if brief.threats:
            lines.append("What would break it: " + "; ".join(f"{t.reason} [{', '.join(t.cites)}]" for t in brief.threats))
    lines.append(f"Evidence: {evidence}. " + " ".join(fields))
    if struck:
        lines.append("Set aside: " + "; ".join(struck) + ".")
    lines.append("Say run to hand off, ask for a picture of anything in the file, or tell me anything to change.")
    _writer()({"convince": {"family": fam.name}})
    return {**update, "reply": "\n".join(ln for ln in lines if ln)}


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
        "artifact": state.get("artifact"),
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
                return Command(goto="handoff", update={"turn": turn, "message": answer, "run_requested": False, "artifact": None})
            return Command(goto="fit", update={"turn": turn, "message": answer, "run_requested": False, "artifact": None})
        # "run" is the person's word that the drafts they were shown stand; empty and refuted fields stay open
        confirmed = []
        for o in state.get("open") or []:
            f = memory.field(o.address)
            if o.status == "drafted" and f is not None and f.value is not None:
                memory.set(o.address, f.value, status="confirmed", source=f"user:turn:{turn}", said=answer)
                confirmed.append(o.address)
        store.save(memory)
        if confirmed:
            record(
                state,
                "claim",
                by="person",
                memory=memory,
                design=design_now(state),
                read=[f"user:turn:{turn}"],
                note="confirmed as drafted: " + ", ".join(confirmed),
            )
        return Command(goto="check", update={"turn": turn, "message": answer, "run_requested": True, "settled_now": confirmed, "artifact": None})
    store.save(memory)
    return Command(goto="infer", update={"turn": turn, "message": answer, "infer_errors": [], "infer_attempts": 0, "settled_now": [], "artifact": None})


# ------------------------------------------------------------------ infer: the Reader over a message, gated by apply


def infer(state: DeskState) -> Command[Literal["infer", "draw", "check", "explain"]]:
    """The Reader over what the person said this turn, source user:turn:<n>: what it settles, confirms, or leaves unknown, and
    whether it asks the desk something or asks for a picture. The story is read into drafts the readback confirms. Three tries
    when a write is refused."""
    memory = F.memory_of(state)
    turn = int(state.get("turn") or 0)
    a = state.get("ask")
    if a is None:
        asked = "(no question was asked; the person spoke freely)"
    elif a.kind == "story":
        asked = f"{RD.STORY} {a.text}\n(settles: " + ", ".join(a.addresses) + ")"
    else:
        asked = a.text + "\n(settles: " + ", ".join(a.addresses) + ")"
    src = f"user:turn:{turn}"
    message = state.get("message") or ""
    before = {ad: (f.value, f.status) for ad, f in memory.fields.items()}
    out, thoughts, rejected = RD.read_words(
        memory, src, message, asked, state.get("open") or [], turn, ask=a, errors=state.get("infer_errors") or [], draft=a is not None and a.kind == "story"
    )
    settled = [ad for ad, f in memory.fields.items() if before.get(ad) != (f.value, f.status)]
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
    asked_draw = (out.draw or "").strip() or state.get("draw_request") or None
    _writer()({"infer": {"settled": settled, "rejected": rejected, "attempt": attempts, **({"focus": focus_update["focus"]} if focus_update else {})}})
    store.save(memory)
    update = {
        "infer_errors": rejected,
        "infer_attempts": attempts,
        "settled_now": settled,
        "debug": thoughts,
        "desk_question": asked_desk,
        "draw_request": asked_draw,
        **focus_update,
    }
    if rejected and attempts < INFER_ATTEMPTS:
        return Command(goto="infer", update=update)
    update["infer_attempts"] = 0
    if settled:
        record(state, "claim", by="person", memory=memory, design=design_now(state), read=[src], note=", ".join(settled))
    if asked_draw:
        return Command(goto="draw", update=update)
    if asked_desk:
        return Command(goto="explain", update={**update, "explain_errors": [], "explain_attempts": 0})
    return Command(goto="check", update=update)


# ------------------------------------------------------------------ draw (the drawing tool), before the run


def draw_context(memory: Memory) -> tuple[str, dict[str, str]]:
    """What the drawing tool is told about the file: the dataset, the change, the beliefs, and one line per column."""
    cols = "\n".join(V.brief_of(memory, c).line() for c in memory.columns.values() if not c.facts.constant)
    return V.context_text(memory) + "\n\nCOLUMNS\n" + cols, {c.key: c.name for c in memory.columns.values()}


def draw(state: DeskState) -> Command[Literal["explain", "check"]]:
    """The person asked for a picture: the drawing tool makes it from the file, the journal records it, and the caption is shown
    before the next thing asked. A picture settles nothing."""
    memory = F.memory_of(state)
    turn = int(state.get("turn") or 0)
    ask = state.get("draw_request") or ""
    context, columns = draw_context(memory)
    req = VD.DrawRequest(dataset=memory.name, moment="pre", memory_version=memory.version, ask=ask, context=context, csv=_csv_path(memory), columns=columns)
    artifact, decline, thoughts = VD.draw(req)
    _writer()({"draw": {"ask": ask, "made": artifact is not None, "why": decline.reason if decline else ""}})
    if artifact is None:
        assert decline is not None
        drawn = f"I could not draw that: {decline.reason}"
        update: dict = {"drawn": drawn, "artifact": None}
    else:
        record(
            state,
            "explore",
            by="model",
            memory=memory,
            design=design_now(state),
            read=[f"user:turn:{turn}"],
            left=[str(VD.store.folder(artifact.dataset, artifact.moment, artifact.design, artifact.id))],
            note=ask,
        )
        update = {"drawn": f"{artifact.caption} [{artifact.address}]", "artifact": artifact.model_dump()}
    update.update({"draw_request": None, "debug": thoughts})
    if state.get("desk_question"):
        return Command(goto="explain", update={**update, "explain_errors": [], "explain_attempts": 0})
    return Command(goto="check", update=update)


# ------------------------------------------------------------------ explain: the Explainer before the run, gated by the cites


MAX_EXPLAIN_ATTEMPTS = 3


def explain(state: DeskState) -> Command[Literal["explain", "draw", "check"]]:
    """The person asked the desk something: the Explainer answers from the families' knowledge, the matrix, the steps and the
    memory, every cite checked by the one gate; three tries, then the honest fallback. Shown before the next thing asked. An
    answer that turns out to be a drawing request goes to the drawing tool."""
    memory = F.memory_of(state)
    st = state.get("status")
    a = state.get("ask")
    question = state.get("desk_question") or ""
    errs = state.get("explain_errors") or []
    matrix = state.get("matrix") if isinstance(state.get("matrix"), Matrix) else None
    mat = X.before_material(memory, matrix, state.get("probes") or [], journal_of(state).steps(), a.text if a else None)
    out, thought = X.answer_from(mat, memory, state.get("exchanges") or [], question, "before", errs)
    problems = X.gate(out, mat, "before")
    attempts = int(state.get("explain_attempts") or 0) + 1
    _writer()({"explain": {"question": question, "attempt": attempts, "problems": problems}})
    if problems and attempts < MAX_EXPLAIN_ATTEMPTS:
        return Command(goto="explain", update={"explain_errors": problems, "explain_attempts": attempts, "debug": [thought]})
    if not problems and out.kind == "draw":
        return Command(
            goto="draw",
            update={"draw_request": (out.draw or "").strip(), "desk_question": None, "explain_errors": [], "explain_attempts": 0, "debug": [thought]},
        )
    if problems:
        text = "I can only answer that from what is settled. " + (_design_line(st, memory, state.get("frame")) if st else "")
    else:
        text = out.text.strip() + (" " + " ".join(f"[{c}]" for c in dict.fromkeys(out.cites) if c not in out.text) if out.cites else "")
    record(
        state,
        "explain",
        by="model",
        memory=memory,
        design=design_now(state),
        read=[] if problems else list(dict.fromkeys(out.cites)),
        note=question + (" (unanswered)" if problems else ""),
    )
    return Command(goto="check", update={"explained": text.strip(), "desk_question": None, "explain_errors": [], "explain_attempts": 0, "debug": [thought]})
