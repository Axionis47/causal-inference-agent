"""The interview before the run: check and probe the memory, the story, the readback, one gap question per turn asked because a
decision needs it, the ready moment, listen, the Reader over what was said, the drawing tool, and the Explainer."""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from causal_agent.common.contracts import QuestionFrame
from causal_agent.desk import edits as ED
from causal_agent.desk import explainer as X
from causal_agent.desk import reader as RD
from causal_agent.desk import turns as TD
from causal_agent.desk.contracts import Ask, Finding, Reading
from causal_agent.desk.nodes import frame as F
from causal_agent.desk.nodes.shared import (
    CAT,
    INFER_ATTEMPTS,
    QUIT_WORDS,
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
    probes_and_facts,
    record,
    thread_id,
    value_words,
)
from causal_agent.desk.readback import compose_readback
from causal_agent.desk.state import Context, DeskState
from causal_agent.explore import client as XC
from causal_agent.families import registry as R
from causal_agent.families.base import Decision, Family
from causal_agent.memory import ops
from causal_agent.memory import views as V
from causal_agent.memory.matrix import Matrix
from causal_agent.memory.records import Memory
from causal_agent.profile import data as PD

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
                    memory.mark(fd.address, "contradiction")
    return {"findings": [Finding.of(f) for f in findings], "refutations": refs, **F.saved(memory)}


def probe_fit(state: DeskState) -> dict:
    memory = F.memory_of(state)
    df = V.table_of(memory)
    fr = state.get("frame")
    needs = focused_needs(state)
    columns = _columns_in_play(memory, fr)
    probes = probes_and_facts(memory, df, fr, columns)
    status = ops.fit(memory, probes, needs, columns=columns, cat=CAT)
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


def _col_ask(field: str, change: str) -> str:
    """What a per-column field asks, in the world's words."""
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
    lines.append(INVITATION + " Say which of these you care about and I will only ask what they need.")
    return "\n".join(lines)


INVITATION = "Tell me more about the data, ask for a picture of anything in it, or say when you want to run."


def standing(status, memory: Memory, frame: QuestionFrame | None) -> str:
    """The reply for a turn with nothing else to say: where the question stands, and what the person can do next."""
    return (_design_line(status, memory, frame) + " " + INVITATION).strip()


def acknowledge(memory: Memory, addresses: list[str]) -> str:
    lines = []
    for a in addresses:
        f = memory.field(a)
        if f is None or f.value is None and f.status != "unknown":
            continue
        lines.append(f"[{a}] " + ("unknown" if f.status == "unknown" else value_words(kind_of(a), Memory.parse(a)[2], f.value)))
    return ("Noted: " + "; ".join(lines) + "\n\n") if lines else ""


def _told_already(memory: Memory, opened: list) -> bool:
    """Whether the story was already told, by a note or by the person in the chat: a draft among the open fields comes from
    doc:<name> or from user:turn:<n>. Asking for it again would be asking for what the memory already holds."""
    for o in opened:
        f = memory.field(o.address)
        if o.status == "drafted" and f is not None and (f.source or "").startswith(("doc:", "user:turn:")):
            return True
    return False


def ask(state: DeskState) -> Command[Literal["listen", "fit"]]:
    """The reply this turn, and whether a question rides with it. Without run intent nothing is asked: the reply is what the desk
    answered, drew or noted, the map once, or where the question stands. With run intent, the story once (unless a note or the
    person told it), then the readback of the drafts, then one blocking gap per turn, and at ready the ready moment every
    time: the routing nodes (fit, decide, gate, design) run and `convince` says what they made; the person's word launches
    from there."""
    st = state["status"]
    memory = F.memory_of(state)
    opened = state.get("open") or []
    intent = bool(state.get("run_intent"))
    flags: dict = {}
    a: Ask | None = None
    if intent and not state.get("story_asked") and opened:
        flags["story_asked"] = True
        story = memory.field("claim:story.text")
        if not _told_already(memory, opened) and (story is None or story.value is None):
            a = Ask(addresses=[o.address for o in opened], kind="story", text=STORY_TEXT, because=sorted({b for o in opened for b in o.because}))
    if a is None and intent:  # only what blocks is asked; a relation a decision rests on never does
        opened = [o for o in opened if not o.optional or o.status == "drafted"]
        a = compose_ask(memory, opened, state.get("findings") or [], state.get("frame"), surviving=list(st.surviving))
        if a is not None and a.kind == "confirm":
            flags["readback_done"] = True
    head = (state.get("note") or "") + acknowledge(memory, state.get("settled_now") or [])
    if not state.get("oriented"):  # the first reply after the question is read: the map of what the file could answer
        head = compose_map(st, memory, state.get("frame")) + "\n\n" + head
    if state.get("explained"):  # the person asked the desk something last turn: the answer comes first
        head = state["explained"] + "\n\n" + head
    if state.get("explored"):  # the person asked to look at the data last turn: the summary comes first, the artifact beside the reply
        head = state["explored"] + "\n\n" + head
    if state.get("transformed"):  # the person asked to change the table last turn: the version now in play comes first
        head = state["transformed"] + "\n\n" + head
    shown = {"artifact": state.get("artifact"), "explored": None, "transformed": None, "oriented": True, "explained": None, "note": "", **flags}
    if a is None:
        if intent and st.ready:
            return Command(goto="fit", update={"ask": None, "reply": head, "convincing": True, "gate_errors": [], "decide_attempts": 0, **shown})
        if intent:
            body = (
                "Nothing more to ask, but no design fits yet: "
                + "; ".join(f"{f} ({w})" for f, w in st.struck.items())
                + ". Tell me what is different about the data, or ask another question."
            )
        else:
            body = "" if head.strip() else standing(st, memory, state.get("frame"))
    else:
        body = ("Before I can run, this still has to be settled. " if intent else "") + a.text
    return Command(goto="listen", update={"ask": a, "reply": (head + body).strip(), **shown})


# ------------------------------------------------------------------ convince (the ready moment)


def after_design(state: DeskState) -> Literal["judge", "handoff"]:
    """Where the design goes: to the judge and then the ready moment when the interview sent the routing nodes to show it, else
    on to the hand-off (the word, or a what-if)."""
    return "judge" if state.get("convincing") else "handoff"


def convince(state: DeskState, runtime: Runtime[Context]) -> dict:
    """The ready moment, said from what the routing nodes made: the decision (by code, or one judgement among several), the
    Designer's brief, and the design in the question's words with the evidence, what it bets on, each decision, the threats,
    and the struck families with one reason each; then the judge's line. The pack it reads back is the preview the judge built."""
    memory = F.memory_of(state)
    d = state.get("decision")
    registry = {f.name: f for f in R.knowledge()}
    head = state.get("reply") or ""
    verdicts = {v.family: v for v in state.get("family_verdicts") or []}
    update: dict = {"convinced_version": memory.version, "convincing": False}
    if d is None or state.get("gate_errors") or d.chosen not in registry:
        text = (
            head
            + "Everything the analysis needs is settled, but no family stands: "
            + "; ".join(f"{v.family} ({next((n.note for n in v.needs if not n.met), v.concern or 'does not fit')})" for v in verdicts.values())
            + ". Tell me what is different about the data."
        )
        return {**update, "reply": text, "handoff": None, "brief": None, "preview": None}
    fam = registry[d.chosen]
    brief = state.get("brief")
    probes = [p for p in state.get("probes") or [] if p.family == fam.name and p.passed is not None]
    evidence = "; ".join(f"{p.detail} [{p.address}]" for p in probes) or "no probe applies"
    fields = [f"[claim:assignment.kind] {memory.value('claim:assignment.kind')}"] + [
        f"[{a}]" for a in ("claim:change.what", "claim:grain.panel") if memory.value(a) is not None
    ]
    struck = [
        f"{v.family}: {next((n.note for n in v.needs if not n.met), v.concern or 'does not fit')}"
        for v in verdicts.values()
        if not v.admissible and v.family != fam.name
    ]
    pack = state.get("preview")
    assert pack is not None  # the judge node built it
    lines = [
        head + "Everything the analysis needs is settled. Here is the context the analysis will run on; correct any line before you say yes.",
        pack.render_summary(),
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
    r = state.get("readiness")
    if r is not None:
        gaps = "; ".join(f"{g.reason} [{g.address}; {', '.join(g.cites)}]" for g in r.gaps)
        lines.append(f"A second reading says {r.verdict.replace('_', ' ')}: {r.summary}" + (f" {gaps}" if gaps else ""))
    lines.append("Say yes to hand off, ask for a picture of anything in the file, or tell me anything to change.")
    _writer()({"convince": {"family": fam.name, "judge": r.verdict if r is not None else None}})
    return {**update, "reply": "\n".join(ln for ln in lines if ln)}


# ------------------------------------------------------------------ listen


def listen(state: DeskState) -> Command[Literal["infer", "__end__"]]:
    """The interrupt before the run. Whatever the person says goes to the Reader; only a quit word ends the conversation here.
    The payload's kind says what the page is looking at: ask when a question rides with the reply, ready at the ready moment,
    chat otherwise."""
    st, a = state["status"], state.get("ask")
    memory = F.memory_of(state)
    payload = {
        "phase": "before",
        "kind": "ask" if a is not None else ("ready" if at_ready_moment(state, memory) else "chat"),
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
    turn = int(state.get("turn") or 0) + 1
    _remember(memory, turn, ", ".join((("lane:" + x) if a.from_lane else x) for x in a.addresses) if a else "", answer)
    return Command(
        goto="infer", update={"turn": turn, "message": answer, "infer_errors": [], "infer_attempts": 0, "settled_now": [], "artifact": None, **F.saved(memory)}
    )


# ------------------------------------------------------------------ infer: the Reader over a message, gated by apply


def at_ready_moment(state: DeskState, memory: Memory) -> bool:
    """The desk has just shown the design it would run, decided on the memory as it stands: a yes here launches."""
    return state.get("ask") is None and state.get("decision") is not None and state.get("convinced_version") == memory.version and bool(state.get("ready"))


def infer(state: DeskState) -> Command[Literal["infer", "explore", "transform", "check", "explain", "handoff"]]:
    """The Reader over what the person said this turn, source user:turn:<n>: the kind of turn first, then what it settles,
    confirms, or leaves unknown, and whether it asks the desk something or asks for a picture. The story is read into drafts
    the readback confirms. Three tries when a reading is refused. A run at the ready moment is the word that launches; a run
    anywhere else turns the interview on."""
    memory = F.memory_of(state)
    turn = int(state.get("turn") or 0)
    a = state.get("ask")
    ready_now = at_ready_moment(state, memory)
    if ready_now:
        asked = "(the desk showed the design it would run and asked for the word to run it; a yes here is run)"
    elif a is None:
        asked = "(no question was asked; the person spoke freely)"
    elif a.kind == "story":
        asked = f"{RD.STORY} {a.text}\n(settles: " + ", ".join(a.addresses) + ")"
    else:
        asked = a.text + "\n(settles: " + ", ".join(a.addresses) + ")"
    src = f"user:turn:{turn}"
    message = state.get("message") or ""
    before = {ad: (f.value, f.status) for ad, f in memory.fields.items()}
    note = ""
    edits = ED.edits_in(message)
    if edits is not None:  # lines of `address = value`, as the handoff view sends them: the person's word on each, no model reads them
        verdict, confidence = "update", 1.0
        out, thoughts = Reading(kind="update"), []
        refused = ops.apply(memory, [ops.Update(address=ad, value=v, status="confirmed", source=src, said=f"{ad} = {v}") for ad, v in edits], CAT)
        rejected = []
        if refused:
            note = "Not written: " + "; ".join(refused) + ".\n\n"
    else:
        # the kind first, cheaply: a confident explore, ask or run never reaches the Reader; an update, or an unsure decision,
        # does, and the Reader decides the kind for itself there
        verdict, confidence = TD.classify(message, asked=a.text if a else None, at_ready=ready_now, last_reply=state.get("reply") or "", node=f"decide:{src}")
    if edits is not None:
        pass
    elif verdict in ("explore", "ask", "run", "transform") and not state.get("infer_errors"):
        out = Reading(
            kind=verdict,
            explore=message if verdict == "explore" else None,
            question=message if verdict == "ask" else None,
            transform=message if verdict == "transform" else None,
        )
        thoughts, rejected = [], []
    else:
        # before the person has asked to run, what they say in passing is drafted for the readback to confirm; an answer to the
        # interview is their word; the story is always read into drafts
        draft = not state.get("run_intent") or (a is not None and a.kind == "story")
        out, thoughts, rejected = RD.read_words(
            memory, src, message, asked, state.get("open") or [], turn, ask=a, errors=state.get("infer_errors") or [], draft=draft
        )
    if a is not None and a.kind == "story" and message.strip():  # the account itself, whole, beside what was read from it
        ops.apply(memory, [ops.Update(address="claim:story.text", value=message, status="confirmed", source=src, said=message, verbatim=True)], CAT)
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
    asked_explore = (out.explore or "").strip() or state.get("explore_request") or None
    asked_transform = (out.transform or "").strip() or state.get("transform_request") or None
    _writer()(
        {
            "infer": {
                "kind": out.kind,
                "decided": verdict,
                "confidence": round(confidence, 3),
                "settled": settled,
                "rejected": rejected,
                "attempt": attempts,
                **({"focus": focus_update["focus"]} if focus_update else {}),
            }
        }
    )
    update = {
        **F.saved(memory),
        "infer_errors": rejected,
        "infer_attempts": attempts,
        "settled_now": settled,
        "debug": thoughts,
        "desk_question": asked_desk,
        "explore_request": asked_explore,
        "transform_request": asked_transform,
        **({"note": note} if note else {}),
        **focus_update,
    }
    if rejected and attempts < INFER_ATTEMPTS:
        return Command(goto="infer", update=update)
    update["infer_attempts"] = 0
    if settled:
        record(state, "claim", by="person", memory=memory, design=design_now(state), read=[src], note=", ".join(settled))
    if out.kind == "run":
        if ready_now:  # the word, on the design just shown
            return Command(goto="handoff", update={**update, "artifact": None})
        return Command(goto="check", update={**update, "run_intent": True})  # the interview asks what blocks, then shows the ready moment
    if asked_transform:
        return Command(goto="transform", update=update)
    if asked_explore:
        return Command(goto="explore", update=update)
    if asked_desk:
        return Command(goto="explain", update={**update, "explain_errors": [], "explain_attempts": 0})
    return Command(goto="check", update=update)


# ------------------------------------------------------------------ explore (the exploration agent), before the run


def explore_context(memory: Memory) -> tuple[str, dict[str, str]]:
    """What the exploration agent is told about the file: the dataset, the change, the beliefs, and one line per column."""
    cols = "\n".join(V.brief_of(memory, c).line() for c in memory.columns.values() if not c.facts.constant)
    return V.context_text(memory) + "\n\nCOLUMNS\n" + cols, {c.key: c.name for c in memory.columns.values()}


def pair_names(state: DeskState, memory: Memory) -> tuple[str | None, str | None]:
    """The outcome and the treatment by the file's own names, as far as the frame and the memory know them."""
    fr = state.get("frame")
    out = memory.column(fr.outcome) if fr is not None and fr.outcome else None
    t = memory.value("claim:assignment.treatment_column") or (fr.cause if fr is not None else None)
    tr = memory.column(str(t)) if t else None
    return (out.name if out else None), (tr.name if tr else None)


def explore(state: DeskState) -> Command[Literal["explain", "check"]]:
    """The person asked to look at the data: the exploration agent makes it from the file, the journal records it, and the
    summary is shown with the artifact beside the reply. What it made settles nothing."""
    memory = F.memory_of(state)
    turn = int(state.get("turn") or 0)
    ask = state.get("explore_request") or ""
    context, columns = explore_context(memory)
    outcome, treatment = pair_names(state, memory)
    artifact, decline, thoughts = XC.explore(
        dataset=memory.name,
        ask=ask,
        context=context,
        columns=columns,
        csv=_csv_path(memory),
        outcome=outcome,
        treatment=treatment,
        frozen=False,
        design=None,
        memory_version=memory.version,
        data_version=memory.data_version,
        thread_id=thread_id(),
    )
    _writer()({"explore": {"ask": ask, "made": artifact is not None, "why": decline.reason if decline else ""}})
    if artifact is None:
        assert decline is not None
        update: dict = {"explored": f"I could not look at that: {decline.reason}", "artifact": None}
    else:
        record(
            state,
            "explore",
            by="model",
            memory=memory,
            design=design_now(state),
            read=[f"user:turn:{turn}"],
            left=[str(XC.VA.folder(artifact.dataset, artifact.id))],
            note=ask,
        )
        update = {"explored": f"{artifact.summary} [{artifact.address}]", "artifact": artifact.model_dump()}
    update.update({"explore_request": None, "debug": thoughts})
    if state.get("desk_question"):
        return Command(goto="explain", update={**update, "explain_errors": [], "explain_attempts": 0})
    return Command(goto="check", update=update)


# ------------------------------------------------------------------ transform (the data agent), before the run


def switch_to(state: DeskState, memory: Memory, version, turn: int) -> str:
    """Put the version the agent wrote in play: the memory re-reads the file, the journal records the person's word, and the
    reply's first line says what is in play now. Nothing reruns on its own."""
    path = XC.VS.table_path(memory.name, version.v)
    assert path is not None
    added, dropped = memory.switch_version(version.v, str(path), PD.profile_for(path))
    _writer()({"version": {"v": version.v, "parent": version.parent, "added": added, "dropped": dropped}})
    record(
        state,
        "version",
        by="person",
        memory=memory,
        design=design_now(state),
        read=[f"user:turn:{turn}", f"version:{version.parent}"],
        left=[str(path.parent)],
        note=f"version {version.v}: {version.summary}",
    )
    return f"Table version {version.v} is now in play: {version.summary} [{version.address}]"


def transform(state: DeskState) -> Command[Literal["explain", "check"]]:
    """The person asked to change the table: the data agent writes a new version beside the one in play, the memory switches to
    it on their word, and the interview goes on over the new table; the design is shown again before anything runs."""
    memory = F.memory_of(state)
    turn = int(state.get("turn") or 0)
    ask = state.get("transform_request") or ""
    context, columns = explore_context(memory)
    outcome, treatment = pair_names(state, memory)
    version, decline, thoughts = XC.transform(
        dataset=memory.name,
        ask=ask,
        context=context,
        columns=columns,
        csv=_csv_path(memory),
        outcome=outcome,
        treatment=treatment,
        frozen=False,
        memory_version=memory.version,
        data_version=memory.data_version,
        thread_id=thread_id(),
    )
    _writer()({"transform": {"ask": ask, "made": version is not None, "why": decline.reason if decline else ""}})
    if version is None:
        assert decline is not None
        update: dict = {"transformed": f"I could not change the table: {decline.reason}"}
    else:
        update = {"transformed": switch_to(state, memory, version, turn), **F.saved(memory)}
    update.update({"transform_request": None, "debug": thoughts})
    if state.get("desk_question"):
        return Command(goto="explain", update={**update, "explain_errors": [], "explain_attempts": 0})
    return Command(goto="check", update=update)


# ------------------------------------------------------------------ explain: the Explainer before the run, gated by the cites


MAX_EXPLAIN_ATTEMPTS = 3


def explain(state: DeskState) -> Command[Literal["explain", "explore", "check"]]:
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
    if not problems and out.kind == "explore":
        return Command(
            goto="explore",
            update={"explore_request": (out.explore or "").strip(), "desk_question": None, "explain_errors": [], "explain_attempts": 0, "debug": [thought]},
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
