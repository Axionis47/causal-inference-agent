"""Views of a memory as the desk, the viz tool and the pack builder read it: a column as a brief, a belief, the person's
words, the columns in play, and the memory as text. Nothing here writes the memory."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from causal_agent.common.contracts import Belief, ColumnBrief, Provenance, QuestionFrame, Role, Said, render_change_text, render_dataset_text
from causal_agent.memory.records import Column, Memory
from causal_agent.profile import datasets as DS

BELIEF_KINDS = ("unobserved", "exclusion", "spillover", "trend_continues", "cutoff_only", "mediator")
_VALUE_FIELD = {
    "unobserved": "exists",
    "exclusion": "exists",
    "spillover": "possible",
    "trend_continues": "believed",
    "cutoff_only": "believed",
    "mediator": "exists",
}


def fields_of(memory: Memory, kind: str) -> dict[str, Any]:
    return memory.values_of(f"claim:{kind}")


def brief_of(memory: Memory, col: Column, role: str | None = None) -> ColumnBrief:
    """A column as the lane reads it: the file's facts beside what is known about it. The role is the caller's view."""
    fs = memory.fields_of(col.address)
    v = {n: f.value for n, f in fs.items() if f.value is not None}
    src = next((fs[n].source for n in ("meaning", "when") if n in v and fs[n].source), None)
    prov = {n: Provenance(status=f.status, source=f.source, said=f.said) for n, f in fs.items() if f.value is not None or f.status != "empty"}
    return ColumnBrief(
        name=col.name,
        key=col.key,
        role=cast(Role, role or "candidate"),
        meaning=v.get("meaning"),
        stands_for=v.get("stands_for"),
        proxy=v.get("proxy"),
        when=v.get("when") or "unknown",
        set_by=v.get("set_by"),
        moved_by_change=v.get("moved_by_change"),
        measures_outcome=v.get("measures_outcome"),
        source=src,
        provenance=prov,
        facts=col.facts,
    )


def belief_of(memory: Memory, kind: str) -> Belief | None:
    fs = {n: f for n, f in memory.fields_of(f"claim:{kind}").items() if f.value is not None or f.status != "empty"}
    if not fs:
        return None
    vf = fs.get(_VALUE_FIELD[kind]) or next(iter(fs.values()))
    known = {n: f.value for n, f in fs.items() if f.value is not None}
    return Belief(
        kind=kind,
        value=known.get(_VALUE_FIELD[kind]),
        what=known.get("what"),
        why=known.get("why") or known.get("why_believed"),
        column=known.get("column"),
        status=vf.status,
        source=vf.source,
        said=vf.said,
    )


def said_of(memory: Memory) -> list[Said]:
    """The person's words, plus the sentence each known field rests on when the transcript does not carry it."""
    out = list(memory.said)
    seen = {(s.turn, s.text) for s in out}
    for address, f in memory.fields.items():
        if f.said and f.source and f.source.startswith("user:turn:"):
            turn = int(f.source.rsplit(":", 1)[1])
            if (turn, f.said) not in seen:
                out.append(Said(turn=turn, about=address, text=f.said))
                seen.add((turn, f.said))
    return sorted(out, key=lambda s: s.turn)


def in_play(memory: Memory, frame: QuestionFrame | None, entry: dict | None = None) -> list[str]:
    """The columns the question, the rule, or the grain name, by the file's own name. Every other column is out of play:
    never asked about, never in the pack."""
    entry = entry or {}
    a, ch, g = memory.values_of("claim:assignment"), memory.values_of("claim:change"), memory.values_of("claim:grain")
    names: list[str] = []
    cands = ([frame.outcome, frame.cause] + [c.column for c in frame.relevant_columns]) if frame else []
    for n in (
        cands
        + [a.get("treatment_column")]
        + list(a.get("depends_on") or [])
        + [a.get("score_column"), ch.get("date_column"), a.get("level_column"), memory.value("claim:exclusion.column"), memory.value("claim:mediator.column")]
        + list(g.get("key_columns") or [])
        + [entry.get("time")]
        + list(entry.get("entity") or [])
    ):
        c = memory.column(n) if n else None
        if c is not None and c.name not in names:
            names.append(c.name)
    return names


def index_records(memory: Memory) -> list[Column]:
    """Columns the frame may see: drop ids and constants deterministically."""
    return [c for c in memory.columns.values() if c.facts.kind != "id" and not c.facts.constant]


def context_text(memory: Memory) -> str:
    """The memory as the routing and the viz judgements read it: the dataset, the change, the beliefs."""
    return "\n\n".join(
        [
            render_dataset_text(memory.name, memory.facts, fields_of(memory, "grain"), fields_of(memory, "sampling"), fields_of(memory, "missing")),
            render_change_text(fields_of(memory, "change"), fields_of(memory, "assignment")),
            "BELIEFS\n" + ("\n".join(b.render() for k in BELIEF_KINDS if (b := belief_of(memory, k)) is not None) or "(none recorded)"),
        ]
    )


def table_of(memory: Memory) -> pd.DataFrame:
    """The file behind a memory, read whole."""
    path = DS.csv_path(memory.name, memory.csv)
    if path is None:
        raise FileNotFoundError(f"no csv is known for {memory.name!r}")
    return pd.read_csv(path)
