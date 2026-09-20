"""The diff-in-diff family's disqualifiers: periods before the change, and the treated group seen in them."""

from __future__ import annotations

import pandas as pd

from causal_agent.memory.claims import ClaimTable, ProbeResult
from causal_agent.memory.probes import periods, pre_period_probe, pre_periods, treated_mask


def probes(df: pd.DataFrame, table: ClaimTable, th: dict) -> list[ProbeResult]:
    fam = "diff_in_diff"
    out = [pre_period_probe(fam, df, table, th)]
    treated = treated_mask(df, table)
    if treated is not None and pre_periods(df, table) is not None:
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
