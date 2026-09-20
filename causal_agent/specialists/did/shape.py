"""From the raw table to the canonical panel pyfixest's DID surface expects. Facts only.

Canonical columns: y, unit, time, treated, post, treat, rel_time, cohort, plus any candidate control.
`treat` is float because pyfixest's resampling code needs it so.
"""

from __future__ import annotations

import pandas as pd

from causal_agent.specialists.did.contracts import Groups, Periods, ShapeFacts

CANON = ["y", "unit", "time", "treated", "post", "treat", "rel_time", "cohort"]


class ShapeError(Exception):
    """A typed stop: the table cannot be put into before-and-after form. `facts` explain why."""

    def __init__(self, reason: str, facts: list[str], fix: str):
        super().__init__(reason)
        self.reason, self.facts, self.fix = reason, facts, fix


def canonical(table: pd.DataFrame, groups: Groups, periods: Periods, outcome: str, candidates: list[str], unit_column: str | None = None,
              cluster_column: str | None = None) -> tuple[pd.DataFrame, ShapeFacts]:
    """unit_column: the entity column for a long panel (from the dataset index); the group column when absent.
    cluster_column: the level the pack says errors cluster at; carried as `cluster` when it is a column of the table."""
    treated = (table[groups.column].astype(str) == str(groups.treated_level)).astype(int)
    if periods.kind == "wide":
        panel, facts = _from_wide(table, treated, periods, candidates)
    else:
        panel, facts = _from_long(table, treated, periods, outcome, candidates, unit_column or groups.column)
    if cluster_column and cluster_column in table.columns and "cluster" not in panel.columns:
        by_row = table[cluster_column].astype(str)
        if periods.kind == "wide":
            panel["cluster"] = pd.concat([by_row, by_row], ignore_index=True).to_numpy()
        else:
            panel["cluster"] = by_row.to_numpy()[panel.index] if len(panel) == len(table) else _cluster_by_unit(table, panel, unit_column or groups.column, cluster_column)
    return panel, facts


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
    before = base.assign(y=table[p.before_column].to_numpy(), time=0)
    after = base.assign(y=table[p.after_column].to_numpy(), time=1)
    long = pd.concat([before, after], ignore_index=True)
    long["post"] = long["time"]
    long["treat"] = (long["treated"] * long["post"]).astype(float)
    long["rel_time"] = long["time"] - 1
    long["cohort"] = long["treated"]  # first treated period is 1 for treated, 0 for never
    facts = _facts(long, kind="wide")
    _guard(facts)
    return long[CANON + keep], facts


def _from_long(table: pd.DataFrame, treated: pd.Series, p: Periods, outcome: str, candidates: list[str], unit_column: str) -> tuple[pd.DataFrame, ShapeFacts]:
    t = p.time_column
    if t not in table.columns:
        raise ShapeError("the time column is missing", [f"{t!r}"], "a column that orders rows in time")
    time = _ordered(table[t])
    if time is None:
        raise ShapeError("the time column does not sort", [f"{t!r} has values that are neither numbers nor dates"], "a numeric or date time column")
    first_post = _parse_like(time, p.first_post)
    if first_post is None or not (time.min() <= first_post <= time.max() + 1):
        raise ShapeError("the first post period is not inside the time range", [f"first_post {p.first_post!r}; range {time.min()} to {time.max()}"], "a change date inside the data's time span")
    df = pd.DataFrame({"y": table[outcome].to_numpy(), "time": time.to_numpy(), "treated": treated.to_numpy()})
    keep = [c for c in candidates if c in table.columns and c not in (t, outcome, unit_column)]
    for c in keep:
        df[c] = table[c].to_numpy()
    if unit_column not in table.columns:
        raise ShapeError("the unit column is missing", [f"{unit_column!r}"], "an entity column that identifies a unit across periods")
    df["unit"] = table[unit_column].astype(str).to_numpy()
    lo = _parse_like(time, p.window_start) if p.window_start else None
    hi = _parse_like(time, p.window_end) if p.window_end else None
    if lo is not None:
        df = df[df["time"] >= lo]
    if hi is not None:
        df = df[df["time"] <= hi]
    df = df.copy()
    df["post"] = (df["time"] >= first_post).astype(int)
    df["treat"] = (df["treated"] * df["post"]).astype(float)
    periods_sorted = sorted(df["time"].unique())
    index = {v: i for i, v in enumerate(periods_sorted)}
    post_index = next((i for i, v in enumerate(periods_sorted) if v >= first_post), len(periods_sorted))
    df["rel_time"] = df["time"].map(index) - post_index
    df["cohort"] = df["treated"] * post_index
    facts = _facts(df, kind="long")
    _guard(facts)
    return df[CANON + keep], facts


# ------------------------------------------------------------------ helpers


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


def _facts(df: pd.DataFrame, *, kind: str) -> ShapeFacts:
    per_unit = df.groupby("unit")["treated"].nunique()
    switching = int((per_unit > 1).sum())
    treated_units = df.groupby("unit")["treated"].max()
    pre = int(df.loc[df["post"] == 0, "time"].nunique())
    post = int(df.loc[df["post"] == 1, "time"].nunique())
    first_treated = df[df["treat"] > 0].groupby("unit")["time"].min()
    cohorts = int(first_treated.nunique()) if len(first_treated) else 0
    return ShapeFacts(
        kind=kind, rows=int(len(df)),
        units_treated=int((treated_units == 1).sum()), units_control=int((treated_units == 0).sum()),
        periods_pre=pre, periods_post=post, cohorts=max(cohorts, 1 if post else 0), switching_units=switching,
        time_values=[str(v) for v in sorted(df["time"].unique())][:40],
    )


def _guard(f: ShapeFacts) -> None:
    if f.periods_pre == 0:
        raise ShapeError("no period before the change", [f"{f.periods_post} post periods, 0 pre"], "an observation of the outcome from before the change")
    if f.periods_post == 0:
        raise ShapeError("no period after the change", [f"{f.periods_pre} pre periods, 0 post"], "an observation of the outcome from after the change")
    if f.units_treated == 0 or f.units_control == 0:
        raise ShapeError("one of the groups has no units", [f"{f.units_treated} treated, {f.units_control} control"], "rows from both a treated and an untreated group")
    if f.switching_units:
        raise ShapeError("units change group over time", [f"{f.switching_units} units switch"], "a group label fixed per unit")
