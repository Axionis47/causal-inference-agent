"""Difference in differences: the outcome by group over time, with the change marked. The periods before the change come
from the same series."""

from __future__ import annotations

import pandas as pd

from causal_agent.common.contracts import Probe
from causal_agent.memory.records import Memory
from causal_agent.profile.data import column
from causal_agent.viz.graph import FigureDecl, PrevizFigure, VizState
from causal_agent.viz.spec import Figure, FigureSpec, Mark, Series


def _periods(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.5:
        return num
    return pd.to_datetime(s, errors="coerce")


def by_group_over_time(
    df: pd.DataFrame,
    outcome: str | None,
    time: str | None,
    treatment: str | None,
    treated_level: str,
    change_period: str | float | None,
    floor: int = 2,
    addresses: list[str] | None = None,
) -> Figure:
    for c, what in ((outcome, "outcome"), (time, "time"), (treatment, "treatment")):
        if c not in df.columns:
            return Figure.refused(f"the {what} column {c!r} is not in the table", "diff_in_diff.by_group_over_time")
    assert outcome is not None and time is not None and treatment is not None
    treated = df[treatment].astype(str) == str(treated_level)
    if treated.sum() == 0 or (~treated).sum() == 0:
        return Figure.refused("one group is empty", "diff_in_diff.by_group_over_time")
    t = _periods(df[time])
    y = pd.to_numeric(df[outcome], errors="coerce")
    if t.isna().all() or y.isna().all():
        return Figure.refused("the time or the outcome column does not read as numbers or dates", "diff_in_diff.by_group_over_time")
    frame = pd.DataFrame({"t": t, "y": y, "g": treated.map({True: "got the change", False: "did not"})}).dropna()
    means = frame.groupby(["g", "t"], observed=True)["y"].agg(["mean", "size"]).reset_index()
    periods = sorted(frame["t"].unique())
    series = []
    for g in ("got the change", "did not"):
        part = means[means["g"] == g].set_index("t")
        series.append(
            Series(
                name=g,
                x=[str(p.date()) if hasattr(p, "date") else float(p) for p in periods],
                y=[float(part["mean"][p]) if p in part.index else None for p in periods],
                n=[int(part["size"].get(p, 0)) for p in periods],
            )
        )
    marks, pre = [], None
    if change_period is not None:
        cp = _periods(pd.Series([change_period])).iloc[0]
        if pd.notna(cp):
            pre = int(sum(1 for p in periods if p < cp))
            marks.append(Mark(kind="vline", at=str(cp.date()) if hasattr(cp, "date") else float(cp), label="the change"))
    probe = Probe(
        family="diff_in_diff",
        name="pre_periods",
        value=None if pre is None else float(pre),
        passed=None if pre is None else pre >= floor,
        detail=f"{pre} distinct periods before the change; floor {floor}" if pre is not None else "change period not settled",
    )
    spec = FigureSpec(
        id=f"by_group_over_time_{outcome}".replace(" ", "_"),
        kind="lines",
        title=f"{outcome} by group over {time}",
        x_label=time,
        y_label=f"mean {outcome}",
        series=series,
        marks=marks,
        note=(f"{pre} periods before the change to compare movement on" if pre is not None else "the change period is not marked"),
        draws_on=list(addresses or []) + [probe.address],
    )
    return Figure(made=True, spec=spec, probe=probe, function="diff_in_diff.by_group_over_time")


# ------------------------------------------------------------------ what the viz tool may choose


def _by_group_over_time(memory: Memory, df: pd.DataFrame, state: VizState, th: dict) -> Figure:
    a, ch = memory.values_of("claim:assignment"), memory.values_of("claim:change")
    return by_group_over_time(
        df,
        column(df, state.get("outcome")),
        column(df, ch.get("date_column")),
        column(df, a.get("treatment_column")),
        str(a.get("treated_level")),
        ch.get("period_value"),
        floor=int(th["probe"]["min_pre_periods"]),
        addresses=["claim:change.date_column", "claim:change.period_value"],
    )


FIGURES = [
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
