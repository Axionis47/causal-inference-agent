"""What the desk's conversation nodes share: the catalogue and the thresholds, the words that end or run, the stream writer,
and the small readers of a memory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langgraph.config import get_config, get_stream_writer

from causal_agent.common.contracts import QuestionFrame
from causal_agent.families import registry as R
from causal_agent.memory import facts as FX
from causal_agent.memory import ops
from causal_agent.memory import views as V
from causal_agent.memory.catalogue import Catalogue, ClaimKind, load_catalogue, load_thresholds
from causal_agent.memory.claims import ProbeResult
from causal_agent.memory.records import COLUMN_KIND, Memory
from causal_agent.profile import datasets as DS

CAT: Catalogue = load_catalogue()
TH: dict = load_thresholds()
INFER_ATTEMPTS = 3
QUIT_WORDS = {"quit", "exit"}


def thread_id() -> str | None:
    """The conversation's thread, so a process started for this turn groups under it; None outside a graph run."""
    try:
        return str(get_config().get("configurable", {}).get("thread_id") or "") or None
    except RuntimeError:
        return None


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
        if k.verbatim:
            continue  # kept by the desk, never read into
        fields = "; ".join(
            f"{name} ({spec.type}{': ' + ', '.join(f'{o} = {spec.about[str(o)]}' if str(o) in spec.about else str(o) for o in spec.options) if spec.options else ''}{', optional' if spec.optional else ''})"
            + (f" = {spec.hint}" if spec.hint else "")
            for name, spec in k.fields.items()
        )
        tag = " [per column]" if k.per_column else " [uncheckable: only on the person's word]" if k.uncheckable else ""
        cues = f" Reading their words: {k.cues}" if k.cues else ""
        out.append(f"- {k.name}{tag}: {k.about}. Fields: {fields}.{cues}")
    return "\n".join(out)


def kind_of(address: str) -> ClaimKind:
    """The catalogue kind an address belongs to: the column kind for col:, the named kind for claim:."""
    return CAT.kinds[Memory.parse(address)[1] if address.startswith("claim:") else COLUMN_KIND]


def options_text(kind: ClaimKind, field: str) -> str:
    """The legal answers of a field, in the world's words: each option with what it means, or yes or no."""
    spec = kind.fields[field]
    if spec.type == "choice":
        return "; ".join(f"{o} ({spec.about[str(o)]})" if str(o) in spec.about else str(o) for o in spec.options)
    if spec.type == "bool":
        return "yes or no"
    if spec.type == "column_or_none":
        return "a column's name, or none"
    return ""


def value_words(kind: ClaimKind, field: str, value) -> str:
    """A field's value as the person reads it: an option with what it means, a list, yes or no."""
    spec = kind.fields.get(field)
    if spec and spec.type == "choice" and str(value) in spec.about:
        return f"{value} ({spec.about[str(value)]})"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _csv_path(memory: Memory) -> Path:
    path = DS.csv_path(memory.name, memory.csv)
    assert path is not None, f"no csv is known for {memory.name!r}"
    return path


def _entry(memory: Memory) -> dict:
    return DS.dataset_entries().get(memory.name) or {}


def _columns_in_play(memory: Memory, frame: QuestionFrame | None) -> list[str]:
    return V.in_play(memory, frame, _entry(memory))


def probes_and_facts(memory: Memory, df: pd.DataFrame, frame: QuestionFrame | None, columns: list[str]) -> list[ProbeResult]:
    """Every family's probes, then the data facts for the columns in play. Both ride in `probes`: the probes carry a verdict a
    family can be struck by, the facts carry numbers the pack shows and the chat and the lane can cite."""
    probes = ops.probe(memory, df, R.REGISTRY.values(), TH, CAT)
    a = memory.values_of("claim:assignment")
    treatment = a.get("treatment_column") or (frame.cause if frame else None)
    outcome = frame.outcome if frame else None
    return probes + FX.facts(df, memory, memory.to_claims(CAT), outcome=outcome, treatment=treatment, columns=columns)


def focused_needs(state) -> dict:
    """The families' needs the interview and the fit read: every family's, or only those of the families the person named."""
    needs = R.needs()
    focus = [f for f in (state.get("focus") or []) if f in needs]
    selected = state.get("context_family")
    if selected in needs and (not focus or selected in focus):
        return {selected: needs[selected]}
    return {k: v for k, v in needs.items() if k in focus} if focus else needs
