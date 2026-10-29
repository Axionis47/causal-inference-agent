"""Pure question composition and prerequisite ordering over endpoint obligations."""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from typing import Literal

from causal_agent.common.contracts import QuestionFrame
from causal_agent.desk.contracts import Ask, Finding
from causal_agent.desk.nodes.shared import (
    CAT,
    options_text,
    value_words,
)
from causal_agent.desk.readback import compose_readback
from causal_agent.families import registry as R
from causal_agent.families.base import Decision, Family
from causal_agent.memory.records import Memory


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
