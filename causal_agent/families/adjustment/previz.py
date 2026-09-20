"""Adjustment: do the arms overlap on what the offer depended on? Shares of each level by arm, and the smallest cell."""

from __future__ import annotations

import pandas as pd

from causal_agent.common.addresses import key as _key
from causal_agent.families.adjustment.overlap import levels_of, overlap_probe
from causal_agent.memory.records import Memory
from causal_agent.profile.data import column
from causal_agent.viz.graph import FigureDecl, PrevizFigure, VizState
from causal_agent.viz.spec import Figure, FigureSpec, Series


def overlap(df: pd.DataFrame, treatment: str | None, treated_level: str, columns: list[str], floor: int = 20, addresses: list[str] | None = None) -> Figure:
    """Share of each level by arm for every column the offer depended on, and the overlap probe from the same cells."""
    if treatment not in df.columns:
        return Figure.refused(f"{treatment!r} is not a column in the table", "adjustment.overlap")
    assert treatment is not None
    columns = [c for c in columns if c in df.columns and c != treatment]
    if not columns:
        return Figure.refused("no column to show overlap on: nothing the offer depended on is named", "adjustment.overlap")
    treated = df[treatment].astype(str) == str(treated_level)
    if treated.sum() == 0 or (~treated).sum() == 0:
        return Figure.refused(f"one arm is empty: {int(treated.sum())} rows have {treatment} = {treated_level!r}", "adjustment.overlap")
    probe = overlap_probe(df, treated, columns, floor)
    x: list[str] = []
    ys: dict[str, list[float]] = {"treated": [], "control": []}
    ns: dict[str, list[int]] = {"treated": [], "control": []}
    for c in columns:
        lv = levels_of(df[c])
        for level in list(dict.fromkeys(lv.sort_values())):
            x.append(f"{c} = {level}")
            for arm, mask in (("treated", treated), ("control", ~treated)):
                n = int(((lv == level) & mask).sum())
                ns[arm].append(n)
                ys[arm].append(float(n / max(int(mask.sum()), 1)))
    spec = FigureSpec(
        id=f"overlap_{'_'.join(_key(c) for c in columns)}",
        kind="bars",
        title=f"Who got the change, by {', '.join(columns)}",
        x_label="level",
        y_label="share of the arm",
        series=[
            Series(name=f"{treatment} = {treated_level}", x=list(x), y=ys["treated"], n=ns["treated"]),
            Series(name=f"{treatment} ≠ {treated_level}", x=list(x), y=ys["control"], n=ns["control"]),
        ],
        note=("both arms appear at every level" if probe.passed else "some level has one arm thin or missing")
        + f"; smallest cell {int(probe.value or 0)} rows",
        draws_on=list(addresses or []) + [probe.address],
    )
    return Figure(made=True, spec=spec, probe=probe, function="adjustment.overlap")


# ------------------------------------------------------------------ what the viz tool may choose


def _overlap(memory: Memory, df: pd.DataFrame, state: VizState, th: dict) -> Figure:
    a = memory.values_of("claim:assignment")
    t = column(df, a.get("treatment_column"))
    named = [c for n in (state["point"].columns or []) if (c := column(df, n)) is not None]
    deps = named or [c for d in (a.get("depends_on") or []) if (c := column(df, d)) is not None]
    return overlap(
        df,
        t,
        str(a.get("treated_level")),
        deps,
        floor=int(th["arms"]["min_rows_cell"]),
        addresses=["claim:assignment.depends_on", "claim:assignment.treatment_column"],
    )


FIGURES = [
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
