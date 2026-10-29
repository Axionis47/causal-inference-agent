"""Outcome-free, score-oriented inputs for optional cutoff diagnostic coding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_agent.common.addresses import key
from causal_agent.explore.diagnostics import DiagnosticTools, tools_for
from causal_agent.families.discontinuity.lane.engine import shape as S
from causal_agent.lane import nodes as L


def validate_sources(h, score, cluster) -> None:
    """Reject outcome aliases before any reasoning tool can inspect their values."""
    outcome = key(h.outcome)
    essential = {score.column, score.takeup_column, cluster} - {None}
    if outcome in essential:
        raise ValueError("the target outcome cannot also supply score, receipt or clustering diagnostic values")
    for role, column in [("score", score.column), ("receipt", score.takeup_column)]:
        card = h.column(column) if column else None
        if card and (card.measures_outcome is True or card.role == "outcome"):
            raise ValueError(f"the declared diagnostic {role} cannot be an outcome measure")
        if role == "score" and card and card.moved_by_change is True:
            raise ValueError("the declared diagnostic score cannot be treatment-affected")
    cluster_card = h.column(cluster) if cluster else None
    if cluster_card and (cluster_card.measures_outcome is True or cluster_card.role == "outcome" or cluster_card.moved_by_change is True):
        raise ValueError("the declared diagnostic cluster cannot be an outcome measure or treatment-affected column")


def diagnostic_table(state) -> tuple[pd.DataFrame, str]:
    """Keep every admitted row independently of whether its withheld outcome is observed."""
    h, score = state["handoff"], state["score"]
    outcome, cluster = key(h.outcome), state.get("cluster_column")
    validate_sources(h, score, cluster)
    essential = {score.column, score.takeup_column, cluster} - {None}
    raw = L.table(state)
    x = pd.to_numeric(raw[score.column], errors="coerce")
    effective, shift = S.effective_cutoff(x, score)
    sign = 1 if score.treated_side == "above" else -1
    oriented = sign * (x - effective)
    side = sign * (x - float(score.cutoff))
    eligible = (side >= 0 if score.cutoff_value_treated else side > 0).astype(float)
    eligible[~np.isfinite(x)] = np.nan
    projected = pd.DataFrame({"x": oriented.where(np.isfinite(x)), "side": eligible})
    mapping = {"x": score.column, "side": "admitted cutoff eligibility"}
    if score.takeup_column:
        projected["t"] = S.receipt_indicator(raw[score.takeup_column], score.takeup_level)
        mapping["t"] = f"{score.takeup_column} equals {score.takeup_level!r}"
    if cluster:
        projected["cluster"] = raw[cluster]
        mapping["cluster"] = cluster
    allowed = set(state["covariates"].balance_tested) | set(state["covariates"].adjusted)
    for role in state["ladder"].covariates.items:
        column = role.column
        card = h.column(column)
        if (
            column in allowed
            and column not in essential | {outcome}
            and column in raw
            and role.predetermined
            and not role.affected_by_treatment
            and not role.is_outcome_measure
            and card is not None
            and card.moved_by_change is not True
            and card.measures_outcome is not True
            and card.role != "outcome"
            and card.facts.kind != "id"
            and not (card.when == "after" and h.status(f"col:{column}.when") == "confirmed")
        ):
            alias = S.covcol(column)
            projected[alias] = raw[column]
            mapping[alias] = column
    metadata = (
        f"DATA_CSV has all {len(projected)} admitted rows, including rows with missing outcomes; no outcome values. "
        f"Columns map to source fields: {mapping!r}. x = {sign} * ({score.column} - {effective!r}); "
        f"original cutoff {score.cutoff!r}, effective-cutoff shift {shift!r}. "
        f"The admitted treated side is {score.treated_side!r}; equality at the original cutoff is "
        f"{'treated' if score.cutoff_value_treated else 'control'}. side is the exact admitted eligibility indicator; "
        "nonfinite scores have missing x and side. The engine retains its existing complete-case analysis population."
    )
    return projected, metadata


def diagnostic_tools(state) -> tuple[DiagnosticTools, str]:
    table, metadata = diagnostic_table(state)
    outcome = key(state["handoff"].outcome)
    dropped_name = None if outcome in table.columns else outcome
    return tools_for(table, Path(state["run_dir"]) / "diagnostics", dropped_name=dropped_name, required=False), metadata
