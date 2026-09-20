"""What every family's probes share: the periods before the change, the unit column, the treated mask, and the runner
that asks each family for its probes. The probes themselves live with their families."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from causal_agent.memory.checks import treated_mask as treated_mask  # shared with the families' probes
from causal_agent.memory.claims import ClaimTable, ProbeResult
from causal_agent.profile.data import column


def periods(df: pd.DataFrame, table: ClaimTable) -> tuple[pd.Series | None, str | None]:
    ch = table.get("change")
    if not ch:
        return None, None
    dc = column(df, ch.fields.get("date_column"))
    return (df[dc] if dc is not None else None), ch.fields.get("period_value")


def pre_periods(df: pd.DataFrame, table: ClaimTable) -> int | None:
    col, pv = periods(df, table)
    if col is None or pv is None:
        return None
    num = pd.to_numeric(col, errors="coerce")
    try:
        cut = float(pv)
        return int(num[num < cut].nunique()) if num.notna().any() else None
    except ValueError:
        pass
    try:
        d, c = pd.to_datetime(col, errors="coerce"), pd.to_datetime(pv)
        return int(d[d < c].dt.to_period("D").nunique()) if d.notna().any() else None
    except Exception:
        return None


def unit_col(df: pd.DataFrame, table: ClaimTable) -> str | None:
    g = table.get("grain")
    keys = (g.fields.get("key_columns") or []) if g else []
    col, _ = periods(df, table)
    for k in keys:
        c = column(df, k)
        if c is not None and (col is None or c != col.name):
            return c
    return None


def run_probes(df: pd.DataFrame, table: ClaimTable, families: Iterable[Any], th: dict) -> list[ProbeResult]:
    """Every family's probes, in the order given. A family is anything with a `probes` callable, or none."""
    out: list[ProbeResult] = []
    for fam in families:
        fn = getattr(fam, "probes", None)
        if fn is not None:
            out.extend(fn(df, table, th))
    return out
