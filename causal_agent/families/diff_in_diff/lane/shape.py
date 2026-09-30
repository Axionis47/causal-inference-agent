"""From the raw table to the canonical panel pyfixest's DID surface expects. Facts only.

Canonical columns: y, unit, time, time_index (the period's 1-based rank), treated (ever treated), post (at or after the
earliest first-treated period, for everyone), treat (treated in this period), rel_time (periods since the unit's own first
treated period; -1, the reference, for a unit never treated), cohort (the unit's first treated period's index; 0 for never
treated, which is what pyfixest's DID estimators call gname), plus any candidate control. `treat` is float because
pyfixest's resampling code needs it so.

Each unit's first treated period is read in this order: an adoption column the pack names (the period each unit first got
the change; empty, zero or outside the span means never); a treatment indicator that switches on within a unit and stays
on; else the treated label and the one change period. One first period for every treated unit is one-shot adoption;
several is staggered, and the facts say which units got the change when.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from causal_agent.families.diff_in_diff.lane.contracts import Groups, Periods, ShapeFacts

CANON = ["y", "unit", "time", "time_index", "treated", "post", "treat", "rel_time", "cohort"]


class ShapeError(Exception):
    """A typed stop: the table cannot be put into before-and-after form. `facts` explain why."""

    def __init__(self, reason: str, facts: list[str], fix: str):
        super().__init__(reason)
        self.reason, self.facts, self.fix = reason, facts, fix


def canonical(
    table: pd.DataFrame,
    groups: Groups,
    periods: Periods,
    outcome: str,
    candidates: list[str],
    unit_column: str | None = None,
    cluster_column: str | None = None,
    cohort_column: str | None = None,
) -> tuple[pd.DataFrame, ShapeFacts]:
    """unit_column: the entity column for a long panel (from the dataset index); the group column when absent.
    cluster_column: the level the pack says errors cluster at; carried as `cluster` when it is a column of the table.
    cohort_column: the pack's adoption column, the period each unit first got the change, when it names one."""
    treated = (table[groups.column].astype(str) == str(groups.treated_level)).astype(int)
    if periods.kind == "wide":
        panel, facts = _from_wide(table, treated, periods, candidates)
    else:
        panel, facts = _from_long(table, treated, periods, outcome, candidates, unit_column or groups.column, cohort_column)
    if cluster_column and cluster_column in table.columns and "cluster" not in panel.columns:
        by_row = table[cluster_column].astype(str)
        if periods.kind == "wide":
            panel["cluster"] = pd.concat([by_row, by_row], ignore_index=True).to_numpy()
        else:
            panel["cluster"] = (
                by_row.to_numpy()[panel.index] if len(panel) == len(table) else _cluster_by_unit(table, panel, unit_column or groups.column, cluster_column)
            )
    clusters = int(panel["cluster"].nunique()) if "cluster" in panel.columns else int(panel["unit"].nunique())
    return panel, facts.model_copy(update={"clusters": clusters})


def _cluster_by_unit(table: pd.DataFrame, panel: pd.DataFrame, unit_column: str, cluster_column: str):
    first = table.groupby(table[unit_column].astype(str))[cluster_column].first().astype(str)
    return panel["unit"].map(first).to_numpy()


def _from_wide(table: pd.DataFrame, treated: pd.Series, p: Periods, candidates: list[str]) -> tuple[pd.DataFrame, ShapeFacts]:
    for c in (p.before_column, p.after_column):
        if c not in table.columns or not pd.api.types.is_numeric_dtype(table[c]):
            raise ShapeError("a before or after column is missing or not numeric", [f"{c!r}"], "two numeric columns holding the outcome before and after")
    keep = [c for c in candidates if c in table.columns and c not in (p.before_column, p.after_column)]
    base = pd.DataFrame({"unit": range(len(table)), "treated": treated.to_numpy()})
    for c in keep:
        base[c] = table[c].to_numpy()
    before = base.assign(y=table[p.before_column].to_numpy(), time=0, time_index=1)
    after = base.assign(y=table[p.after_column].to_numpy(), time=1, time_index=2)
    long = pd.concat([before, after], ignore_index=True)
    long["post"] = long["time"]
    long["treat"] = (long["treated"] * long["post"]).astype(float)
    long["cohort"] = 2 * long["treated"]  # the after period is the first treated one; 0 for never treated
    long["rel_time"] = np.where(long["treated"] == 1, long["time_index"] - 2, -1)
    facts = _facts(long, kind="wide", source="label", labels={1: "before", 2: "after"})
    _guard(facts)
    return long[CANON + keep], facts


def _from_long(
    table: pd.DataFrame, treated: pd.Series, p: Periods, outcome: str, candidates: list[str], unit_column: str, cohort_column: str | None
) -> tuple[pd.DataFrame, ShapeFacts]:
    t = p.time_column
    if t not in table.columns:
        raise ShapeError("the time column is missing", [f"{t!r}"], "a column that orders rows in time")
    time = _ordered(table[t])
    if time is None:
        raise ShapeError("the time column does not sort", [f"{t!r} has values that are neither numbers nor dates"], "a numeric or date time column")
    first_post = _parse_like(time, p.first_post)
    if first_post is None or not (time.min() <= first_post <= time.max() + 1):
        raise ShapeError(
            "the first post period is not inside the time range",
            [f"first_post {p.first_post!r}; range {time.min()} to {time.max()}"],
            "a change date inside the data's time span",
        )
    if unit_column not in table.columns:
        raise ShapeError("the unit column is missing", [f"{unit_column!r}"], "an entity column that identifies a unit across periods")
    df = pd.DataFrame(
        {"y": table[outcome].to_numpy(), "time": time.to_numpy(), "treated": treated.to_numpy(), "unit": table[unit_column].astype(str).to_numpy()}
    )
    keep = [c for c in candidates if c in table.columns and c not in (t, outcome, unit_column)]
    for c in keep:
        df[c] = table[c].to_numpy()
    lo = _parse_like(time, p.window_start) if p.window_start else None
    hi = _parse_like(time, p.window_end) if p.window_end else None
    if lo is not None:
        df = df[df["time"] >= lo]
    if hi is not None:
        df = df[df["time"] <= hi]
    df = df.copy()
    periods_sorted = sorted(df["time"].unique())
    index = {v: i + 1 for i, v in enumerate(periods_sorted)}  # 1-based: pyfixest's gname reads 0 as never treated
    labels = {i: period_label(v) for v, i in index.items()}
    df["time_index"] = df["time"].map(index).astype(int)
    post_index = next((index[v] for v in periods_sorted if v >= first_post), len(periods_sorted) + 1)
    first, source = _first_treated(df, table, table[unit_column].astype(str), time, index, post_index, cohort_column)
    gnum = df["unit"].map(first).fillna(0).astype(int)
    df["cohort"] = gnum
    df["treated"] = (gnum > 0).astype(int)
    earliest = int(gnum[gnum > 0].min()) if (gnum > 0).any() else post_index
    df["post"] = (df["time_index"] >= earliest).astype(int)
    df["treat"] = ((gnum > 0) & (df["time_index"] >= gnum)).astype(float)
    df["rel_time"] = np.where(gnum > 0, df["time_index"] - gnum, -1).astype(int)
    facts = _facts(df, kind="long", source=source, labels=labels)
    _guard(facts)
    return df[CANON + keep], facts


def _first_treated(
    df: pd.DataFrame, table: pd.DataFrame, unit_of_row: pd.Series, time: pd.Series, index: dict, post_index: int, cohort_column: str | None
) -> tuple[pd.Series, Literal["label", "indicator", "cohort_column"]]:
    """Each unit's first treated period as a 1-based index, 0 for never: the pack's adoption column, else an indicator that
    switches on within a unit, else the treated label and the one change period."""
    if cohort_column and cohort_column in table.columns:
        per_unit = table[cohort_column].groupby(unit_of_row).first().dropna()
        out: dict[str, int] = {}
        bad: list[str] = []
        for u, v in per_unit.items():
            if pd.isna(v) or (isinstance(v, (int, float, np.integer, np.floating)) and float(v) == 0.0):
                out[str(u)] = 0
                continue
            parsed = _parse_like(time, v)
            if parsed is None:
                bad.append(str(u))
                continue
            later = [i for p, i in index.items() if p >= parsed]
            out[str(u)] = min(later) if later else 0  # an adoption after the span: never treated within it
        if bad:
            raise ShapeError("an adoption period could not be read", [f"{len(bad)} units, such as {bad[0]!r}"], "adoption periods written like the time column")
        return pd.Series(out), "cohort_column"
    per_unit_levels = df.groupby("unit")["treated"].nunique()
    if (per_unit_levels > 1).any():
        ordered = df.sort_values(["unit", "time_index"])
        reversals = ordered.groupby("unit")["treated"].apply(lambda s: bool((s.diff() < 0).any()))
        n = int(reversals.sum())
        if n:
            raise ShapeError("units leave the treatment", [f"{n} units switch off after switching on"], "a treatment that stays on once it starts")
        first = ordered[ordered["treated"] == 1].groupby("unit")["time_index"].min()
        return first.reindex(df["unit"].unique()).fillna(0).astype(int), "indicator"
    ever = df.groupby("unit")["treated"].max()
    return (ever * post_index).astype(int), "label"


# ------------------------------------------------------------------ helpers


def period_label(v) -> str:
    """A period as the reader knows it: a whole number without a trailing .0, a date as the file wrote it."""
    if isinstance(v, (float, np.floating)) and np.isfinite(v) and float(v) == int(v):
        return str(int(v))
    if hasattr(v, "date") and hasattr(v, "hour"):
        return str(v.date()) if (v.hour, v.minute, v.second) == (0, 0, 0) else str(v)
    return str(v)


def _ordered(s: pd.Series) -> pd.Series | None:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    try:
        return pd.to_datetime(s)
    except Exception:
        return None


def _parse_like(time: pd.Series, value) -> object | None:
    if value is None:
        return None
    try:
        if pd.api.types.is_numeric_dtype(time):
            return float(value)
        return pd.to_datetime(value)
    except Exception:
        return None


def _facts(df: pd.DataFrame, *, kind: str, source: str, labels: dict[int, str]) -> ShapeFacts:
    gnum = df.groupby("unit")["cohort"].first()
    treated_units = gnum[gnum > 0]
    earliest = int(treated_units.min()) if len(treated_units) else None
    idx = df["time_index"]
    pre = int(idx[idx < earliest].nunique()) if earliest is not None else int(idx.nunique())
    post = int(idx[idx >= earliest].nunique()) if earliest is not None else 0
    by_cohort = {labels.get(int(k), str(k)): int(v) for k, v in treated_units.value_counts().sort_index().items()}
    present = df.groupby("unit")["time_index"].nunique()
    return ShapeFacts(
        kind=kind,
        rows=int(len(df)),
        units_treated=int(len(treated_units)),
        units_control=int((gnum == 0).sum()),
        units_never_treated=int((gnum == 0).sum()),
        never_treated_exists=bool((gnum == 0).any()),
        periods_pre=pre,
        periods_post=post,
        cohorts=int(treated_units.nunique()),
        units_by_cohort=by_cohort,
        adoption="staggered" if treated_units.nunique() > 1 else "one_shot",
        first_treated_source=source,
        treatment_reversals=0,
        balanced=bool(len(present) and present.nunique() == 1 and int(present.iloc[0]) == int(idx.nunique())),
        time_values=[period_label(v) for v in sorted(df["time"].unique())][:40],
    )


def _guard(f: ShapeFacts) -> None:
    if f.periods_pre == 0:
        raise ShapeError("no period before the change", [f"{f.periods_post} post periods, 0 pre"], "an observation of the outcome from before the change")
    if f.periods_post == 0:
        raise ShapeError("no period after the change", [f"{f.periods_pre} pre periods, 0 post"], "an observation of the outcome from after the change")
    if f.units_treated == 0:
        raise ShapeError("no unit got the change", [f"{f.units_control} untreated units"], "rows from units that got the change")
    if not f.never_treated_exists and f.cohorts < 2:
        raise ShapeError(
            "no unit is untreated in any period after the change",
            [f"{f.units_treated} treated units, all from the same period, and none never treated"],
            "never-treated units, or units that got the change later",
        )
