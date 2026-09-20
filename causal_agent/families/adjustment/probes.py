"""The adjustment family's disqualifiers: both arms present, and both arms present in every cell the offer looked at."""

from __future__ import annotations

import pandas as pd

from causal_agent.families.adjustment.overlap import overlap_probe
from causal_agent.memory.claims import ClaimTable, ProbeResult
from causal_agent.memory.probes import treated_mask
from causal_agent.profile.data import column


def probes(df: pd.DataFrame, table: ClaimTable, th: dict) -> list[ProbeResult]:
    fam = "adjustment"
    a = table.get("assignment")
    treated = treated_mask(df, table)
    if treated is None or not a:
        return [ProbeResult(family=fam, name="arms", passed=None, detail="treatment column not settled")]
    out = []
    n_t, n_o = int(treated.sum()), int((~treated).sum())
    floor = int(th["arms"]["min_rows_arm"])
    out.append(
        ProbeResult(
            family=fam,
            name="arms",
            value=float(min(n_t, n_o)),
            passed=min(n_t, n_o) >= floor,
            detail=f"{n_t} treated rows and {n_o} others; floor {floor} an arm",
        )
    )
    deps = [c for d in (a.fields.get("depends_on") or []) if (c := column(df, d)) is not None]
    if deps:  # the same cells the pre-viz draws: both arms present at every level the offer looked at
        ov = overlap_probe(df, treated, deps, int(th["arms"]["min_rows_cell"]))
        out.append(ProbeResult(family=fam, name=ov.name, value=ov.value, passed=ov.passed, detail=ov.detail))
    return out
