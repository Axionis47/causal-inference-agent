"""The overlap table: rows per arm in every cell of the columns the offer looked at. The adjustment family's probe and its
pre-run figure draw the same cells."""

from __future__ import annotations

import pandas as pd

from causal_agent.common.contracts import Probe

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
