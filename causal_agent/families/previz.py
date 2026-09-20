"""Each family's pre-run figures: the declaration the viz judgement reads (what it shows, when it makes the point, what it
needs settled) beside the function that draws it from the memory and the table. The viz tool chooses among a family's
declared figures; it never names one."""

from __future__ import annotations

import pandas as pd

from causal_agent.memory.records import Memory
from causal_agent.profile.data import column
from causal_agent.viz.graph import FigureDecl, PrevizFigure, VizState
from causal_agent.viz.previz import adjustment as PA
from causal_agent.viz.previz import diff_in_diff as PD
from causal_agent.viz.previz import discontinuity as PR
from causal_agent.viz.spec import Figure


def _overlap(memory: Memory, df: pd.DataFrame, state: VizState, th: dict) -> Figure:
    a = memory.values_of("claim:assignment")
    t = column(df, a.get("treatment_column"))
    named = [c for n in (state["point"].columns or []) if (c := column(df, n)) is not None]
    deps = named or [c for d in (a.get("depends_on") or []) if (c := column(df, d)) is not None]
    return PA.overlap(
        df,
        t,
        str(a.get("treated_level")),
        deps,
        floor=int(th["arms"]["min_rows_cell"]),
        addresses=["claim:assignment.depends_on", "claim:assignment.treatment_column"],
    )


def _by_group_over_time(memory: Memory, df: pd.DataFrame, state: VizState, th: dict) -> Figure:
    a, ch = memory.values_of("claim:assignment"), memory.values_of("claim:change")
    return PD.by_group_over_time(
        df,
        column(df, state.get("outcome")),
        column(df, ch.get("date_column")),
        column(df, a.get("treatment_column")),
        str(a.get("treated_level")),
        ch.get("period_value"),
        floor=int(th["probe"]["min_pre_periods"]),
        addresses=["claim:change.date_column", "claim:change.period_value"],
    )


def _density(memory: Memory, df: pd.DataFrame, state: VizState, th: dict) -> Figure:
    a = memory.values_of("claim:assignment")
    return PR.density(
        df,
        column(df, a.get("score_column")),
        float(a["cutoff"]),
        floor=int(th["cutoff"]["min_rows_side"]),
        addresses=["claim:assignment.score_column", "claim:assignment.cutoff"],
    )


def _outcome_by_bin(memory: Memory, df: pd.DataFrame, state: VizState, th: dict) -> Figure:
    a = memory.values_of("claim:assignment")
    return PR.outcome_by_bin(
        df,
        column(df, a.get("score_column")),
        float(a["cutoff"]),
        column(df, state.get("outcome")),
        floor=int(th["cutoff"]["min_rows_side"]),
        addresses=["claim:assignment.score_column", "claim:assignment.cutoff"],
    )


ADJUSTMENT = [
    PrevizFigure(
        FigureDecl(
            name="adjustment.overlap",
            family="adjustment",
            shows="the share of each arm at every level of what the offer depended on, side by side, with the smallest cell counted",
            makes_the_point_when="the point is about overlap, balance, common support, or whether both arms exist at every level",
            needs=["claim:assignment.treatment_column", "claim:assignment.treated_level", "claim:assignment.depends_on"],
        ),
        _overlap,
    )
]

DIFF_IN_DIFF = [
    PrevizFigure(
        FigureDecl(
            name="diff_in_diff.by_group_over_time",
            family="diff_in_diff",
            shows="the mean outcome per period for the units that got the change and the rest, with the change marked",
            makes_the_point_when="the point is about movement before the change, parallel paths, or when the groups part",
            needs=["outcome", "claim:change.date_column", "claim:change.period_value", "claim:assignment.treatment_column", "claim:assignment.treated_level"],
        ),
        _by_group_over_time,
    )
]

DISCONTINUITY = [
    PrevizFigure(
        FigureDecl(
            name="discontinuity.density",
            family="discontinuity",
            shows="how many rows sit in each score bin either side of the cutoff",
            makes_the_point_when="the point is about bunching, manipulation of the score, or whether rows exist close to the line on both sides",
            needs=["claim:assignment.score_column", "claim:assignment.cutoff"],
        ),
        _density,
    ),
    PrevizFigure(
        FigureDecl(
            name="discontinuity.outcome_by_bin",
            family="discontinuity",
            shows="the mean outcome per score bin either side of the cutoff",
            makes_the_point_when="the point is about the jump at the line, or about the outcome being smooth through it",
            needs=["outcome", "claim:assignment.score_column", "claim:assignment.cutoff"],
        ),
        _outcome_by_bin,
    ),
]
