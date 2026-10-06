"""The readback: the drafts read back to the person, grouped by the five claims every decision rests on, each line in
the world's terms with the sentence it rests on. One confirm turn; a yes confirms every draft shown, a correction is
read by the Reader like any other message."""

from __future__ import annotations

from causal_agent.desk.contracts import Ask, AskItem
from causal_agent.desk.nodes.shared import CAT, value_words
from causal_agent.memory.records import Memory

GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Who got the change, and how that was decided", ("assignment",)),
    ("What the change was, and when", ("change",)),
    ("What one row is, and which rows are in the file", ("grain", "sampling", "missing")),
    ("What each column records, and when it was set", ("col",)),
    ("What the design would bet on", ("belief",)),
]
CLOSE = "Is this right? Correct any line, or say yes."


def _group_of(kind: str, per_column: bool, belief: bool) -> str:
    key = "col" if per_column else "belief" if belief else kind
    return next((title for title, kinds in GROUPS if key in kinds), "Also read")


def _rests_on(memory: Memory, address: str) -> str:
    f = memory.field(address)
    if f is None:
        return ""
    if f.said:
        return f' — "{f.said.strip()}"'
    return f" — from {f.source}" if f.source else ""


def _as_text(value) -> str:
    """A draft as the text the memory accepts back through `address = value`: yes or no, a list joined, else its words."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return "" if value is None else str(value)


def _options(memory: Memory, kind, field: str) -> list[str]:
    """Every legal answer when the field has a fixed set: the catalogue's options, yes or no, the file's columns."""
    spec = kind.fields.get(field)
    if spec is None:
        return []
    if spec.type == "choice":
        return [str(o) for o in spec.options]
    if spec.type == "bool":
        return ["yes", "no"]
    if spec.type in ("column", "column_or_none"):
        return [c.name for c in memory.columns.values()] + (["none"] if spec.type == "column_or_none" else [])
    return []


def items_of(memory: Memory, drafts: list) -> list[AskItem]:
    """One item per draft, in the order read back: what the page's questionnaire answers line by line."""
    out = []
    for o in drafts:
        kind = CAT.kinds[o.kind]
        f = memory.field(o.address)
        value = _as_text(f.value if f else None)
        if kind.per_column:
            col = memory.column(o.address[4:].rsplit(".", 1)[0])
            label = f"'{col.name if col else o.address}' {o.field.replace('_', ' ')}"
        else:
            label = f"{o.kind.replace('_', ' ')}, {o.field.replace('_', ' ')}"
        out.append(AskItem(address=o.address, label=label, value=value, options=_options(memory, kind, o.field), recommended=value or None))
    return out


def compose_readback(memory: Memory, opened: list) -> Ask | None:
    """The drafts among `opened`, grouped and read back, as one confirm turn; None when nothing is drafted. The page gets the
    same drafts as items, each with the desk's draft recommended and the other legal answers, so a line can be changed on its
    own (sent back as `address = value`, the person's word) while a yes confirms every line."""
    drafts = [o for o in opened if o.status == "drafted"]
    if not drafts:
        return None
    grouped: dict[str, list[str]] = {}
    seen_cols: dict[str, list] = {}
    for o in drafts:
        kind = CAT.kinds[o.kind]
        title = _group_of(o.kind, kind.per_column, kind.uncheckable)
        if kind.per_column:
            seen_cols.setdefault(o.address.rsplit(".", 1)[0], []).append(o)
            grouped.setdefault(title, [])
            continue
        f = memory.field(o.address)
        line = f"{o.kind.replace('_', ' ')}, {o.field.replace('_', ' ')}: {value_words(kind, o.field, f.value if f else None)}" + _rests_on(memory, o.address)
        grouped.setdefault(title, []).append(line)
    col_title = _group_of("measured", True, False)
    for prefix, fields in seen_cols.items():
        col = memory.column(prefix[4:])
        name = col.name if col else prefix
        kind = CAT.kinds[fields[0].kind]
        parts = []
        for o in fields:
            f = memory.field(o.address)
            words = value_words(kind, o.field, f.value if f else None)
            if o.field == "when":
                words = {"before": "fixed before the change", "at": "set at the change", "after": "measured after the change"}.get(
                    str(f.value if f else ""), words
                )
            elif o.field == "meaning":
                words = f"records {words}"
            else:
                words = f"{o.field.replace('_', ' ')}: {words}"
            parts.append(words)
        rests = next((_rests_on(memory, o.address) for o in fields if _rests_on(memory, o.address)), "")
        grouped.setdefault(col_title, []).append(f"'{name}': " + "; ".join(parts) + rests)
    blocks = []
    for title, _ in GROUPS + [("Also read", ())]:
        lines = grouped.get(title)
        if lines:
            blocks.append(title + ":\n" + "\n".join(f"• {ln}" for ln in lines))
    text = "Here is what I read, and what each line rests on.\n\n" + "\n\n".join(blocks) + "\n\n" + CLOSE
    return Ask(
        addresses=[o.address for o in drafts],
        kind="confirm",
        text=text,
        options=["Yes, all right", "No"],
        recommended="Yes, all right",
        items=items_of(memory, drafts),
        because=sorted({b for o in drafts for b in o.because}),
    )
