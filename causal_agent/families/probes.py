"""Each family's disqualifiers, computed once assignment and change are settled. pandas only, no fits. A failed probe
strikes the family out; a probe that cannot run leaves it in. Probes never pick a family."""

from __future__ import annotations

import pandas as pd

from causal_agent.memory.claims import ClaimTable, ProbeResult
from causal_agent.memory.probes import pre_period_probe, treated_mask, unit_col
from causal_agent.profile.data import column


def _single_unit_family(fam: str, df: pd.DataFrame, table: ClaimTable, th: dict) -> list[ProbeResult]:
    out = [pre_period_probe(fam, df, table, th)]
    treated, unit = treated_mask(df, table), unit_col(df, table)
    if treated is not None and unit is not None:
        n_units = int(df.loc[treated, unit].nunique())
        cap = int(th["probe"]["max_treated_units_single"])
        out.append(
            ProbeResult(
                family=fam,
                name="treated_units",
                value=float(n_units),
                passed=1 <= n_units <= cap,
                detail=f"{n_units} treated units by {unit!r}; needs one or at most {cap}",
            )
        )
    return out


def synthetic_control(df: pd.DataFrame, table: ClaimTable, th: dict) -> list[ProbeResult]:
    return _single_unit_family("synthetic_control", df, table, th)


def interrupted_series(df: pd.DataFrame, table: ClaimTable, th: dict) -> list[ProbeResult]:
    return _single_unit_family("interrupted_series", df, table, th)


def instrument(df: pd.DataFrame, table: ClaimTable, th: dict) -> list[ProbeResult]:
    fam = "instrument"
    ex = table.get("exclusion")
    if not ex or ex.status in {"empty", "refuted", "unknown"} or ex.fields.get("exists") is not True:
        return []
    ic = column(df, ex.fields.get("column"))
    if ic is None:
        return [ProbeResult(family=fam, name="instrument_column", passed=False, detail=f"{ex.fields.get('column')!r} is not a column in the file")]
    n = int(df[ic].nunique())
    return [ProbeResult(family=fam, name="instrument_varies", value=float(n), passed=n >= 2, detail=f"{ic!r} takes {n} values")]
