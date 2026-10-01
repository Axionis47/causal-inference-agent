"""The diff-in-diff family's disqualifiers: periods before the change, the treated group seen in them, and how many units got
the change."""

from __future__ import annotations

import pandas as pd

from causal_agent.memory.claims import ClaimTable, ProbeResult
from causal_agent.memory.probes import periods, pre_period_probe, pre_periods, treated_mask, unit_col


def probes(df: pd.DataFrame, table: ClaimTable, th: dict) -> list[ProbeResult]:
    fam = "diff_in_diff"
    out = [pre_period_probe(fam, df, table, th)]
    treated = treated_mask(df, table)
    if treated is None:
        return out
    unit = unit_col(df, table)
    n_units = int(df.loc[treated, unit].nunique()) if unit else int(treated.sum())
    out.append(
        ProbeResult(
            family=fam,
            name="treated_units",
            value=float(n_units),
            passed=n_units > 0,
            detail=f"{n_units} distinct units got the change" if unit else f"{n_units} treated rows; no unit column is settled, so rows stand for units",
        )
    )
    if pre_periods(df, table) is not None:
        col, pv = periods(df, table)
        if col is None or pv is None:
            return out
        num = pd.to_numeric(col, errors="coerce")
        try:
            before = num < float(pv)
            n_tb = int((treated & before).sum())
            out.append(
                ProbeResult(
                    family=fam, name="treated_before", value=float(n_tb), passed=n_tb > 0, detail=f"{n_tb} rows of the treated group observed before the change"
                )
            )
        except (TypeError, ValueError):
            pass
    return out
