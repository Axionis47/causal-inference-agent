"""The Reader: words into claims with reasons. One prompt over a message or a note; one call, gated by `ops.apply`.

The source tag decides what the words may do. `user:turn:<n>` confirms: the person's word settles a field, confirms a
draft, or marks one unknown. `doc:<name>` drafts: a note never confirms, never marks unknown, and never sets a belief;
the gate refuses a belief from a note anyway, and the filter here spares a retry. The caller decides whether to ask
again on a rejection."""

from __future__ import annotations

from causal_agent.common import trace
from causal_agent.common.contracts import Thought
from causal_agent.common.llm import structured
from causal_agent.desk.contracts import Ask, Reading
from causal_agent.desk.nodes.shared import CAT, kind_of, kinds_text, options_text
from causal_agent.desk.prompts import journey as P
from causal_agent.memory import ops
from causal_agent.memory import views as V
from causal_agent.memory.records import COLUMN_KIND, Memory

DESCRIPTION = "(no question: this is the description)"
STORY = "(the story)"


def open_lines(memory: Memory, opened: list, asked: Ask | None) -> str:
    """The open fields as the Reader sees them: the ones just asked first, each with what it asks and its legal values."""
    lines = []
    order = (asked.addresses if asked else []) + [o.address for o in opened if not asked or o.address not in asked.addresses]
    by = {o.address: o for o in opened}
    for a in order:
        o = by.get(a)
        if o is None:
            kind = kind_of(a)
            field = Memory.parse(a)[2]
            spec = kind.fields.get(field) if field else None
            lines.append(f"{a} · {spec.hint if spec and spec.hint else kind.frame} · {options_text(kind, field) if spec and field else ''}")
            continue
        kind = CAT.kinds[o.kind]
        spec = kind.fields[o.field]
        lines.append(f"{a} · {spec.hint or kind.frame} · {options_text(kind, o.field) or spec.type}" + (" (optional)" if o.optional else ""))
    return "\n".join(lines) or "(nothing open)"


def kind_errors(out: Reading, *, from_note: bool) -> list[str]:
    """The kind gate, code: only update carries writes; explore and ask carry what they are about; a note is read as update."""
    errors = []
    if out.kind != "update" and (out.updates or out.confirms or out.unknown):
        errors.append(f"kind is {out.kind} but the reading carries writes; a message that says anything about the data is update")
    if out.kind == "explore" and not (out.explore or "").strip():
        errors.append("kind is explore but explore is empty; say what to look at, in the person's words")
    if out.kind == "ask" and not (out.question or "").strip():
        errors.append("kind is ask but question is empty; say what was asked, in the person's words")
    if out.kind == "transform" and not (out.transform or "").strip():
        errors.append("kind is transform but transform is empty; say what to change in the table, in the person's words")
    if from_note and out.kind != "update":
        errors.append(f"kind is {out.kind} but a note is always read as update")
    return errors


def _belief(address: str) -> bool:
    where, name, _ = Memory.parse(address)
    kind = CAT.kinds.get(COLUMN_KIND if where == "col" else name)
    return kind is not None and kind.uncheckable


def read_words(
    memory: Memory,
    source: str,
    material: str,
    asked: str,
    opened: list,
    turn: int,
    *,
    ask: Ask | None = None,
    errors: list[str] | None = None,
    draft: bool = False,
) -> tuple[Reading, list[Thought], list[str]]:
    """One reading of `material` tagged `source`, written through the gate. Returns the reading, the thoughts, and why each
    write was refused. `asked` is the question the words answer, or DESCRIPTION or STORY; `ask` orders the open fields
    it settles first. `errors` are the last call's refusals, shown so the model fixes them. With `draft`, or from a note,
    the updates are written as drafts for the readback to confirm; confirms and unknowns stay the person's word."""
    from_note = source.startswith("doc:")
    errs = errors or []
    shown = ("\nPREVIOUS READING WAS REJECTED:\n" + "\n".join(f"- {e}" for e in errs) + "\nFix it and return the reading again.\n") if errs else ""
    cols = "\n".join(V.brief_of(memory, c).line() for c in memory.columns.values() if not c.facts.constant)
    user = P.READ_USER.format(
        kinds=kinds_text(),
        memory=memory.render() or "(nothing known yet)",
        asked=asked,
        open=open_lines(memory, opened, ask),
        columns=cols,
        source=source,
        material=material,
        errors=shown,
    )
    out, thought = structured(Reading, P.READ_SYSTEM, user, node=f"read:{source}", tier="light")
    wrong = trace.gate(kind_errors(out, from_note=from_note), name="gate:kind", kind=out.kind)
    if wrong:
        return out, [thought], wrong
    if out.kind != "update":  # nothing to write: the kind gate has held that the reading carries no writes
        return out, [thought], []
    status = "drafted" if from_note or draft else "confirmed"
    updates = []
    for u in out.updates:
        if from_note and _belief(u.address):
            continue  # a note cannot state a belief
        updates.append(ops.Update(address=u.address, value=u.value, status=status, source=source, said=u.said or material[:200], reason=u.reason))
    if not from_note:
        for addr in out.confirms:
            f = memory.field(addr)
            if f is not None and f.value is not None and f.status in {"drafted", "refuted"}:
                updates.append(ops.Update(address=addr, value=f.value, status="confirmed", source=source, said=material[:200], reason="confirmed as drafted"))
        for addr in out.unknown:
            updates.append(ops.Update(address=addr, status="unknown", source=source, said=material[:200]))
    # a refused reading writes nothing: the updates go to a copy first, and only a reading the gate passes whole reaches the
    # memory, so a re-prompted reading never finds its own earlier drafts there to "confirm" (a real run did exactly that)
    rejected = trace.gate(ops.apply(memory.snapshot(), updates, CAT), name="gate:writes", updates=[u.address for u in updates])
    if not rejected:
        ops.apply(memory, updates, CAT)
    return out, [thought], rejected
