"""Family disqualifiers computed once assignment and change are settled. pandas only, no fits. A failed probe
strikes a family out; a probe that cannot run leaves it in. Probes never pick a family."""

from __future__ import annotations

import pandas as pd

from causal_agent.memory.checks import treated_mask as _treated_mask
from causal_agent.memory.claims import ClaimTable, ProbeResult
from causal_agent.profile.data import column
from causal_agent.viz.previz.adjustment import overlap_probe


def _periods(df: pd.DataFrame, table: ClaimTable) -> tuple[pd.Series | None, str | None]:
    ch = table.get("change")
    if not ch:
        return None, None
    dc = column(df, ch.fields.get("date_column"))
    return (df[dc] if dc is not None else None), ch.fields.get("period_value")


def _pre_periods(df: pd.DataFrame, table: ClaimTable) -> int | None:
    col, pv = _periods(df, table)
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


def _unit_col(df: pd.DataFrame, table: ClaimTable) -> str | None:
    g = table.get("grain")
    keys = (g.fields.get("key_columns") or []) if g else []
    col, _ = _periods(df, table)
    for k in keys:
        c = column(df, k)
        if c is not None and (col is None or c != col.name):
            return c
    return None


def run_probes(df: pd.DataFrame, table: ClaimTable, families: list[str], th: dict) -> list[ProbeResult]:
    out: list[ProbeResult] = []
    a = table.get("assignment")
    kind = a.fields.get("kind") if a else None
    treated = _treated_mask(df, table)
    pre = _pre_periods(df, table)
    unit = _unit_col(df, table)

    for fam in families:
        if fam == "discontinuity":
            if kind != "cutoff_rule" or not a:
                continue
            sc = column(df, a.fields.get("score_column"))
            if sc is None or a.fields.get("cutoff") is None:
                out.append(ProbeResult(family=fam, name="rows_by_side", passed=None, detail="score column or cutoff not settled"))
                continue
            x = pd.to_numeric(df[sc], errors="coerce")
            c = float(a.fields["cutoff"])
            n_above, n_below = int((x > c).sum()), int((x < c).sum())
            floor = int(th["cutoff"]["min_rows_side"])
            ok = min(n_above, n_below) >= floor
            out.append(
                ProbeResult(
                    family=fam,
                    name="rows_by_side",
                    value=float(min(n_above, n_below)),
                    passed=ok,
                    detail=f"{n_below} rows below {c} and {n_above} above on {sc!r}; floor {floor} a side",
                )
            )
        elif fam == "adjustment":
            if treated is None:
                out.append(ProbeResult(family=fam, name="arms", passed=None, detail="treatment column not settled"))
                continue
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
        elif fam in {"diff_in_diff", "synthetic_control", "interrupted_series"}:
            if pre is None:
                out.append(ProbeResult(family=fam, name="pre_periods", passed=None, detail="period column or change period not settled"))
            else:
                floor = int(th["probe"]["min_pre_periods"])
                out.append(
                    ProbeResult(
                        family=fam, name="pre_periods", value=float(pre), passed=pre >= floor, detail=f"{pre} distinct periods before the change; floor {floor}"
                    )
                )
            if fam == "diff_in_diff" and treated is not None and pre is not None:
                col, pv = _periods(df, table)
                num = pd.to_numeric(col, errors="coerce")
                try:
                    before = num < float(pv)
                    n_tb = int((treated & before).sum())
                    out.append(
                        ProbeResult(
                            family=fam,
                            name="treated_before",
                            value=float(n_tb),
                            passed=n_tb > 0,
                            detail=f"{n_tb} rows of the treated group observed before the change",
                        )
                    )
                except (TypeError, ValueError):
                    pass
            if fam in {"synthetic_control", "interrupted_series"} and treated is not None and unit is not None:
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
        elif fam == "instrument":
            ex = table.get("exclusion")
            if not ex or ex.status in {"empty", "refuted", "unknown"} or ex.fields.get("exists") is not True:
                continue
            ic = column(df, ex.fields.get("column"))
            if ic is None:
                out.append(ProbeResult(family=fam, name="instrument_column", passed=False, detail=f"{ex.fields.get('column')!r} is not a column in the file"))
            else:
                out.append(
                    ProbeResult(
                        family=fam,
                        name="instrument_varies",
                        value=float(df[ic].nunique()),
                        passed=df[ic].nunique() >= 2,
                        detail=f"{ic!r} takes {df[ic].nunique()} values",
                    )
                )
    return out
