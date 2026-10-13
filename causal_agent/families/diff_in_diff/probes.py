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


def context_facts(df: pd.DataFrame, table: ClaimTable, th: dict) -> dict[str, bool | None]:
    """Assignment support from metadata only, using the lane's identical adoption mapping."""
    from causal_agent.families.diff_in_diff.lane.contracts import Groups, Periods
    from causal_agent.families.diff_in_diff.lane.shape import ShapeError, canonical
    from causal_agent.profile.data import column

    assignment, change = table.get("assignment"), table.get("change")
    if assignment is None:
        return {"claim:assignment.never_treated_exists": None}
    treatment = column(df, assignment.fields.get("treatment_column"))
    level = assignment.fields.get("treated_level")
    if treatment is None or level is None:
        return {"claim:assignment.never_treated_exists": None}
    clock = column(df, change.fields.get("date_column")) if change else None
    cohort = column(df, change.fields.get("adoption_column")) if change else None
    unit = unit_col(df, table) if clock else None
    if clock and not unit:
        return {"claim:assignment.never_treated_exists": None}
    names = list(dict.fromkeys(n for n in [treatment, clock, cohort, unit] if n))
    metadata = df[names].copy()
    synthetic = "_assignment_support_outcome"
    while synthetic in metadata:
        synthetic += "_"
    metadata[synthetic] = 0.0
    group = Groups(column=treatment, treated_level=str(level), reason="assignment support from data", cites=[], by="pack")
    if clock:
        assert change is not None
        first = change.fields.get("period_value")
        if first is None:
            eligible = metadata[cohort] != 0 if cohort else metadata[treatment].astype(str) == str(level)
            if not eligible.any():
                return {"claim:assignment.never_treated_exists": None}
            first = metadata.loc[eligible, clock].min()
        periods_ = Periods(kind="long", time_column=clock, first_post=str(first), reason="declared clock", cites=[], by="pack")
    else:
        periods_ = Periods(kind="wide", before_column=synthetic, after_column=synthetic + "_after", reason="assignment membership", cites=[], by="pack")
        metadata[periods_.after_column] = 0.0
    try:
        _, facts = canonical(metadata, group, periods_, synthetic, [], unit_column=unit, cohort_column=cohort, validate_assignment_support=False)
    except (ShapeError, ValueError, KeyError, TypeError):
        return {"claim:assignment.never_treated_exists": None}
    return {"claim:assignment.never_treated_exists": facts.never_treated_exists}
