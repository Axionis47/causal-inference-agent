"""Reading the question against the memory: load it, mine the attached document once when the memory is bare, skim the
columns when the table is wide, and frame the question. One judgement per model node; the memory is written only through
`ops.apply`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pandas as pd
from langgraph.config import get_stream_writer
from langgraph.types import Send

from causal_agent.common.addresses import key as _key
from causal_agent.common.contracts import PrefilterVote, QuestionFrame, render_change_text, render_dataset_text
from causal_agent.common.llm import structured
from causal_agent.desk import handoff as H
from causal_agent.desk.prompts import routing as P
from causal_agent.desk.state import PrefilterTask, RouteState
from causal_agent.memory import ops, store
from causal_agent.memory.catalogue import load_catalogue
from causal_agent.memory.claims import Extraction
from causal_agent.memory.records import Memory
from causal_agent.profile import datasets as DS

MINE_ATTEMPTS = 2


def _writer():
    try:
        return get_stream_writer()
    except Exception:  # outside a graph run
        return lambda *_: None


def memory_of(state: RouteState) -> Memory:
    """The memory on disk, or the what-if fork while one is being routed and run."""
    fork = state.get("fork")
    if fork is not None:
        return fork
    return store.memory_for(state["dataset"])


def table_of(memory: Memory) -> pd.DataFrame:
    csv = memory.csv or (DS.dataset_entries().get(memory.name) or {}).get("csv")
    path = Path(csv) if csv and Path(csv).is_absolute() else Path(DS.ROOT) / csv
    return pd.read_csv(path)


def width_budget() -> int:
    return int(os.getenv("FRAME_WIDTH_BUDGET", os.getenv("ROUTER_WIDTH_BUDGET", "150")))


def index_records(memory: Memory) -> list:
    """Columns the frame may see: drop ids and constants deterministically."""
    return [c for c in memory.columns.values() if c.facts.kind != "id" and not c.facts.constant]


def context_text(memory: Memory) -> str:
    return "\n\n".join([
        render_dataset_text(memory.name, memory.facts, H.fields_of(memory, "grain"), H.fields_of(memory, "sampling"), H.fields_of(memory, "missing")),
        render_change_text(H.fields_of(memory, "change"), H.fields_of(memory, "assignment")),
        "BELIEFS\n" + ("\n".join(b.render() for k in H.BELIEF_KINDS if (b := H.belief_of(memory, k)) is not None) or "(none recorded)"),
    ])


# ------------------------------------------------------------------ load and mine


def load(state: RouteState) -> dict:
    memory_of(state)  # raises for an unknown dataset
    return {"prefilter_votes": [], "family_verdicts": [], "probes": [], "fit_status": None, "debug": [], "gate_errors": [], "decide_attempts": 0}


def _bare(memory: Memory) -> bool:
    """Nothing from a person, a document, or a model yet: only what the file settled on its own."""
    return not any(f.value is not None and f.source and not f.source.startswith(("data", "code:")) for f in memory.fields.values())


def _doc(memory: Memory) -> tuple[str, str] | None:
    """The attached document, when the dataset has one: today the note file in the index entry."""
    e = DS.dataset_entries().get(memory.name) or {}
    if e.get("note") and (Path(DS.ROOT) / e["note"]).exists():
        return "context", (Path(DS.ROOT) / e["note"]).read_text()
    return None


def mine(state: RouteState) -> dict:
    """Once, when the memory holds nothing but the file's facts and a document is attached: read it into drafts, source doc:<name>.
    A description drafts; it never confirms, and it never sets a belief."""
    memory = memory_of(state)
    doc = _doc(memory)
    if not _bare(memory) or doc is None:
        return {}
    from causal_agent.desk.nodes.journey import kinds_text
    from causal_agent.desk.prompts import journey as JP

    name, text = doc
    cat = load_catalogue()
    cards = "\n".join(H.brief_of(memory, c).line() for c in memory.columns.values())
    errors = ""
    debug, rejected = [], []
    for _ in range(MINE_ATTEMPTS):
        user = JP.EXTRACT_USER.format(kinds=kinds_text(), claims=memory.to_claims(cat).render(), cards=cards, asked="(none: this is the description)",
                                      source=f"doc:{name}", material=text, errors=errors)
        out, thought = structured(Extraction, JP.EXTRACT_SYSTEM, user, node="mine")
        debug.append(thought)
        updates = []
        for up in out.updates:
            kind = cat.kinds.get(up.kind)
            if kind is None or kind.uncheckable or up.unknown:
                continue
            for fv in up.values:
                address = f"col:{_key(up.column)}.{fv.name}" if kind.per_column and up.column else f"claim:{up.kind}.{fv.name}"
                updates.append(ops.Update(address=address, value=fv.value, status="drafted", source=f"doc:{name}", reason=up.reason))
        rejected = ops.apply(memory, updates, cat)
        if not rejected:
            break
        errors = "\nPREVIOUS UPDATES WERE REJECTED:\n" + "\n".join(f"- {e}" for e in rejected) + "\nFix them and return the full set again.\n"
    store.save(memory)
    _writer()({"mine": {"drafted": memory.version, "rejected": rejected}})
    return {"debug": debug}


# ------------------------------------------------------------------ prefilter (wide tables only) and frame


def fan_out_prefilter(state: RouteState) -> list[Send] | Literal["frame"]:
    memory = memory_of(state)
    recs = index_records(memory)
    if len(recs) <= width_budget():
        return "frame"
    changes = render_change_text(H.fields_of(memory, "change"), H.fields_of(memory, "assignment"))
    return [Send("prefilter", PrefilterTask(question=state["question"], changes=changes, column=c.name, card=H.brief_of(memory, c).render())) for c in recs]


def prefilter(task: PrefilterTask) -> dict:
    vote, thought = structured(PrefilterVote, P.PREFILTER_SYSTEM, P.PREFILTER_USER.format(question=task["question"], changes=task["changes"], card=task["card"]),
                               node=f"prefilter:{task['column']}")
    vote.column = task["column"]
    return {"prefilter_votes": [vote], "debug": [thought]}


def frame(state: RouteState) -> dict:
    memory = memory_of(state)
    recs = index_records(memory)
    votes = {v.column: v.relevant for v in state.get("prefilter_votes", [])}
    if votes:
        recs = [c for c in recs if votes.get(c.name, True)]
    index = "\n".join(H.brief_of(memory, c).line() for c in recs)
    fr, thought = structured(QuestionFrame, P.FRAME_SYSTEM, P.FRAME_USER.format(question=state["question"], digest=context_text(memory), column_index=index), node="frame")
    normalise_columns(fr, memory)
    _writer()({"frame": {"intent": fr.intent, "outcome": fr.outcome, "cause": fr.cause, "relevant": [c.column for c in fr.relevant_columns]}})
    return {"frame": fr, "debug": [thought]}


def normalise_columns(fr: QuestionFrame, memory: Memory) -> None:
    """The model may name columns by key (math_score). Downstream wants the file's own name (math score)."""
    for group in (fr.outcome_candidates, fr.cause_candidates, fr.relevant_columns):
        for cand in group:
            rec = memory.column(cand.column)
            if rec is not None:
                cand.column = rec.name
    fr.relevant_columns = [c for c in fr.relevant_columns if memory.column(c.column) is not None]
