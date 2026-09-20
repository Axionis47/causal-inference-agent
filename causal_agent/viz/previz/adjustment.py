"""Adjustment: do the arms overlap on what the offer depended on? Shares of each level by arm, and the smallest cell."""

from __future__ import annotations

import pandas as pd

from causal_agent.common.addresses import key as _key
from causal_agent.common.contracts import Probe
from causal_agent.viz.spec import Figure, FigureSpec, Series

MAX_LEVELS = 6


def levels_of(s: pd.Series, max_levels: int = MAX_LEVELS) -> pd.Series:
    """A column as a small set of levels: its values when few, quantile bins when numeric and many."""
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > max_levels:
        q = pd.qcut(s, q=max_levels, duplicates="drop")
        return q.astype(str)
    return s.astype(str)


def cells(df: pd.DataFrame, treated: pd.Series, columns: list[str]) -> pd.DataFrame:
    """Rows per arm in every combination of the columns' levels: the overlap table."""
    if not columns:
        return pd.DataFrame({"treated": [int(treated.sum())], "control": [int((~treated).sum())]})
    lv = pd.concat([levels_of(df[c]).rename(c) for c in columns], axis=1)
    lv["_arm"] = treated.map({True: "treated", False: "control"})
    t = lv.groupby(columns + ["_arm"], observed=True).size().unstack("_arm", fill_value=0)
    for arm in ("treated", "control"):
        if arm not in t.columns:
            t[arm] = 0
    return t[["treated", "control"]]


def overlap_probe(df: pd.DataFrame, treated: pd.Series, columns: list[str], floor: int) -> Probe:
    """The smallest arm in any cell, against the floor. Every cell holding both arms above the floor is overlap."""
    t = cells(df, treated, columns)
    smallest = int(t.min(axis=1).min()) if len(t) else 0
    empty = int((t.min(axis=1) == 0).sum())
    detail = (
        f"{len(t)} cells over {', '.join(columns) or 'no columns'}; smallest arm in a cell {smallest}; "
        f"{empty} cell{'s' if empty != 1 else ''} with one arm missing; floor {floor}"
    )
    return Probe(family="adjustment", name="overlap", value=float(smallest), passed=smallest >= floor, detail=detail)


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
