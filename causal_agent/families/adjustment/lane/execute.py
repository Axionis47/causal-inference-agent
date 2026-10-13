"""The adjustment lane's execution: one worker per contrast runs the frozen design and nothing else: the estimate, its
falsifications, its sensitivity analyses, the secondary estimator, the estimates within each modifier's levels. It reads the
Design and the table; it never reads the catalogue or the reasoning's state. A fit that fails goes back to the pick once."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from langgraph.types import Command, Send

from causal_agent.common.contracts import Estimate, Refutation
from causal_agent.families.adjustment.lane.artifacts import fan_out_interpret
from causal_agent.families.adjustment.lane.contracts import Design
from causal_agent.families.adjustment.lane.engine import estimators, falsify, sensitivity
from causal_agent.families.adjustment.lane.engine import model as engine
from causal_agent.families.adjustment.lane.state import ContrastTask, SpecialistState, latest_primary
from causal_agent.lane import nodes as L
from causal_agent.lane.nodes import MAX_PICK_ATTEMPTS

_writer, _stop = L.writer, L.stop


# ------------------------------------------------------------------ analyse (fact, fan-out per contrast)


def fan_out_analyse(state: SpecialistState):
    return [Send("analyse", ContrastTask(contrast=c.key, design=state["design"].model_dump(), table_path=state["table_path"])) for c in state["contrasts"]]


def analyse(task: ContrastTask) -> dict:
    d = Design.model_validate(task["design"])
    c = next(x for x in d.contrasts if x.key == task["contrast"])
    table = pd.read_csv(task["table_path"])
    sub = engine.contrast_table(table, d.graph.treatment, c)
    model = engine.build_model(sub, d.graph, d.graph.outcome)
    estimates: list[Estimate] = []
    refutations: list[Refutation] = []
    primary, bundle = estimators.estimate(model, d.estimator, c.key, d.target_units, frozen=d.estimand)
    estimates.append(primary)
    if primary.error is None:
        refutations += [
            falsify.run(model, bundle, primary, rf, c.key) if rf.kind == "falsification" else sensitivity.run(model, bundle, rf, c.key) for rf in d.refuters
        ]
    if d.also_run is not None:
        sec, _ = estimators.estimate(model, d.also_run, c.key, d.target_units, frozen=d.estimand, secondary=True)
        estimates.append(sec)
    if primary.error is None:
        estimates += _by_modifier(sub, d, c.key)
    _writer()({"analyse": {c.key: {"estimate": primary.model_dump(exclude_none=True), "refutations": [r.detail for r in refutations]}}})
    out: dict[str, Any] = {"estimates": estimates, "refutations": refutations}
    if primary.error and primary.error.startswith("design:"):
        out["mismatches"] = {c.key: primary.error}
    return out


def _modifier_groups(sub: pd.DataFrame, column: str, max_levels: int) -> list[tuple[str, pd.DataFrame]]:
    """The rows of each level of a modifier: its values when few, quantile bins when it is a number with many, the most common
    levels otherwise."""
    s = sub[column]
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > max_levels:
        bins = pd.qcut(s, q=max_levels, duplicates="drop")
        return [(str(level), sub[bins == level]) for level in bins.cat.categories]
    levels = [v for v in s.astype(str).value_counts().index[:max_levels]]
    return [(str(v), sub[s.astype(str) == v]) for v in levels]


def _by_modifier(sub: pd.DataFrame, d: Design, contrast_key: str) -> list[Estimate]:
    """The primary estimator run again within each level of each modifier, on the design frozen for that modifier. Too few rows in
    an arm is recorded as the estimate's error, never skipped in silence."""
    floor = d.within_rules.floor
    out: list[Estimate] = []
    entry = d.estimator
    for col, w in d.within.items():
        if col not in sub.columns:
            continue
        g = w.graph
        for level, part in _modifier_groups(sub, col, d.within_rules.max_levels):
            n_t, n_c = int((part[engine.TREATED] == 1).sum()), int((part[engine.TREATED] == 0).sum())
            if min(n_t, n_c) < floor:
                out.append(
                    Estimate(
                        contrast=contrast_key,
                        method=entry.name,
                        n_treated=n_t,
                        n_control=n_c,
                        target_units=d.target_units,
                        modifier=col,
                        level=level,
                        error=f"too few rows in an arm within this level ({n_t} treated, {n_c} control; floor {floor})",
                    )
                )
                continue
            e, _ = estimators.estimate(engine.build_model(part.drop(columns=[col]), g, g.outcome), entry, contrast_key, d.target_units, frozen=w.estimand)
            out.append(e.model_copy(update={"modifier": col, "level": level}))
    return out


def after_analyse(state: SpecialistState) -> Command[Literal["interpret", "pick_estimator", "figures", "feasibility"]]:
    mismatches = state.get("mismatches") or {}
    if mismatches:  # the engine did not identify what the design froze: a stop, never a re-pick
        return _stop(
            "analyse",
            "the engine identified a different set than the design froze",
            [f"{c}: {why}" for c, why in mismatches.items()],
            "a design frozen on the graph the engine runs",
        )
    failed = [e for e in latest_primary(state).values() if e.error]
    if not failed:
        sends = fan_out_interpret(state)
        return Command(goto=sends or "figures")
    if state.get("pick_attempts", 0) < MAX_PICK_ATTEMPTS:
        _writer()({"fit_failure": [e.error for e in failed]})
        return Command(goto="pick_estimator", update={"excluded_estimators": (state.get("excluded_estimators") or []) + [state["estimator"]]})
    return _stop(
        "estimate",
        "the estimator failed to fit and the re-pick failed too",
        [f"{e.contrast}: {e.error}" for e in failed],
        "a different estimator entry or a smaller adjustment set",
    )
