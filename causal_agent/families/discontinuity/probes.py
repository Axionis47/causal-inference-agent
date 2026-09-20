"""The discontinuity family's disqualifier: rows on both sides of the cutoff, enough of them."""

from __future__ import annotations

import pandas as pd

from causal_agent.memory.claims import ClaimTable, ProbeResult
from causal_agent.profile.data import column


def probes(df: pd.DataFrame, table: ClaimTable, th: dict) -> list[ProbeResult]:
    fam = "discontinuity"
    a = table.get("assignment")
    if not a or a.fields.get("kind") != "cutoff_rule":
        return []
    sc = column(df, a.fields.get("score_column"))
    if sc is None or a.fields.get("cutoff") is None:
        return [ProbeResult(family=fam, name="rows_by_side", passed=None, detail="score column or cutoff not settled")]
    x = pd.to_numeric(df[sc], errors="coerce")
    c = float(a.fields["cutoff"])
    n_above, n_below = int((x > c).sum()), int((x < c).sum())
    floor = int(th["cutoff"]["min_rows_side"])
    ok = min(n_above, n_below) >= floor
    return [
        ProbeResult(
            family=fam,
            name="rows_by_side",
            value=float(min(n_above, n_below)),
            passed=ok,
            detail=f"{n_below} rows below {c} and {n_above} above on {sc!r}; floor {floor} a side",
        )
    ]
