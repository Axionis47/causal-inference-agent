"""Shared stream, typed stops and read-only table tools. Judgements remain in their family."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
from langgraph.config import get_stream_writer
from langgraph.types import Command

from causal_agent.common.addresses import key as _key
from causal_agent.common.contracts import Feasibility
from causal_agent.lane.state import LaneState
from causal_agent.lane.tools import Tools


def writer() -> Callable[[Any], None]:
    try:
        return get_stream_writer()
    except Exception:  # outside a graph run
        return lambda _x: None


def question(state: LaneState) -> str:
    return state.get("question") or ""


def stop(stage: str, reason: str, facts: list[str], fix: str, extra: dict | None = None, *, terminal_kind="unsupported_method") -> Command:
    """The honest stop: a Feasibility record and the jump to the feasibility node."""
    f = Feasibility(terminal_kind=terminal_kind, stage=stage, reason=reason, facts=facts, what_would_fix=fix)
    return Command(goto="feasibility", update={"feasibility": f, **(extra or {})})


def data_tools(
    state: LaneState,
    treated: pd.Series | None,
    *,
    table_: pd.DataFrame | None = None,
    aliases: Mapping[str, str] | None = None,
    allow_outcome_rows: pd.Series | None = None,
    allow_outcome_words: str | None = None,
) -> Tools:
    """The read-only data tools over this run's table, with the pair named so no tool joins them, the treated rows when the
    arms are settled, and every column's timing as the pack settled it. A lane may hand in its own canonical table instead
    (a panel, a recentred score) with `aliases` from the pack's names to its columns, and the rows on which the outcome may be
    joined with the arms before the freeze, when its design has such rows."""
    h = state["handoff"]
    assert h is not None
    t, y, _ = keys(state)
    timing = {k: (b.when if (b := h.column(k)) is not None else "unknown") for k in state.get("columns", {})}
    df = table(state) if table_ is None else table_
    al = dict(aliases or {})
    outcome = al.get(y, y) if y else y
    treatment = al.get(t, t) if t else t
    tools = Tools(df, outcome=outcome, treatment=treatment, treated=treated, timing_of=timing, allow_outcome_rows=allow_outcome_rows, aliases=al or None)
    if allow_outcome_words:
        tools.allow_outcome_words = allow_outcome_words
    return tools


def keys(state: LaneState) -> tuple[str | None, str, list[str]]:
    """The treatment (None when the change is the rule itself), the outcome, and every other loaded column."""
    h = state["handoff"]
    assert h is not None
    t = _key(h.treatment) if h.treatment else None
    y = _key(h.outcome)
    return t, y, [k for k in state.get("columns", {}) if k not in (t, y)]


def rejected(errors: list[str] | None) -> str:
    """Show a bounded judgement the concrete reasons its previous record failed."""
    return ("\nPREVIOUS ANSWER WAS REJECTED\n" + "\n".join(f"- {e}" for e in errors) + "\n") if errors else ""


def table(state: LaneState) -> pd.DataFrame:
    return pd.read_csv(state["table_path"])


def feasibility(state: LaneState) -> dict:
    f = state["feasibility"]
    assert f is not None
    writer()({"feasibility": f.model_dump()})
    return {}
