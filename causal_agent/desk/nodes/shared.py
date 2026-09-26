"""What the desk's conversation nodes share: the catalogue and the thresholds, the words that end or run, the stream writer,
and the small readers of a memory."""

from __future__ import annotations

from pathlib import Path

from langgraph.config import get_stream_writer

from causal_agent.common.contracts import QuestionFrame, Said
from causal_agent.families import registry as R
from causal_agent.memory import views as V
from causal_agent.memory.catalogue import Catalogue, load_catalogue, load_thresholds
from causal_agent.memory.records import Memory
from causal_agent.profile import datasets as DS

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
    path = DS.csv_path(memory.name, memory.csv)
    assert path is not None, f"no csv is known for {memory.name!r}"
    return path


def _entry(memory: Memory) -> dict:
    return DS.dataset_entries().get(memory.name) or {}


def _columns_in_play(memory: Memory, frame: QuestionFrame | None) -> list[str]:
    return V.in_play(memory, frame, _entry(memory))


def focused_needs(state) -> dict:
    """The families' needs the interview and the fit read: every family's, or only those of the families the person named."""
    needs = R.needs()
    focus = [f for f in (state.get("focus") or []) if f in needs]
    return {k: v for k, v in needs.items() if k in focus} if focus else needs


def _remember(memory: Memory, turn: int, about: str, text: str) -> None:
    if text and not any(s.turn == turn for s in memory.said):
        memory.said.append(Said(turn=turn, about=about, text=text))


def _design_line(status, memory: Memory, frame: QuestionFrame | None) -> str:
    fams = status.surviving
    if not fams:
        return ""
    ch = memory.values_of("claim:change")
    return f"The question, {frame.outcome if frame else 'the outcome'} against {ch.get('what') or 'the change'}, can be answered by {', '.join(f.replace('_', ' ') for f in fams)}."
