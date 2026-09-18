"""Deterministic profiler: one CSV in, one Profile out. No model, no judgement.

Same file and same arguments always give the same JSON. It describes the file's
shape and reports anything odd. It never says what a column means or what
analysis to run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from causal_agent.common.addresses import key as _key

PROFILER_VERSION = "0.1.0"

Kind = Literal["id", "numeric", "categorical", "boolean", "datetime", "text"]
VariesOver = Literal["entity", "time", "both", "neither", "unknown"]

PLACEHOLDER_STRINGS = {"", "na", "n/a", "nan", "null", "none", "?", "-", "unknown", "missing"}
PLACEHOLDER_NUMBERS = {-1, -9, -99, -999, -9999, 99, 999, 9999, 99999, 999999}
BOOL_TOKENS = {"0", "1", "true", "false", "yes", "no", "y", "n", "t", "f"}


class Sentinel(BaseModel):
    value: str
    count: int
    reason: str


class NumericStats(BaseModel):
    min: float
    max: float
    mean: float
    p01: float
    p25: float
    p50: float
    p75: float
    p99: float


class TopValue(BaseModel):
    value: str
    count: int
    share: float


class DatetimeStats(BaseModel):
    first: str
    last: str
    inferred_frequency: str | None


class SwitchProfile(BaseModel):
    """For columns that vary within an entity: how they turn on and off."""

    entities_that_switch: int
    entities_never_on: int
    entities_always_on: int
    switches_per_entity_median: float
    first_on: str | None  # earliest date any entity turns on
    last_entity_first_on: str | None  # date the last entity to turn on did so


class ColumnProfile(BaseModel):
    name: str
    key: str
    kind: Kind
    nulls: int
    null_rate: float
    distinct: int
    constant: bool
    numeric: NumericStats | None = None
    top_values: list[TopValue] | None = None
    datetime: DatetimeStats | None = None
    observed_sentinels: list[Sentinel] = Field(default_factory=list)
    varies_over: VariesOver = "unknown"
    switch: SwitchProfile | None = None
    format_issues: list[str] = Field(default_factory=list)


class TimeCoverage(BaseModel):
    column: str
    first: str
    last: str
    inferred_frequency: str | None
    gaps: int


class EntitySummary(BaseModel):
    columns: list[str]
    entities: int
    rows_per_entity_min: int
    rows_per_entity_median: float
    rows_per_entity_max: int


class DatasetProfile(BaseModel):
    profiler_version: str
    file: str
    file_hash: str
    rows: int
    columns: int
    duplicate_rows: int
    candidate_keys: list[list[str]]
    grain: list[str] | None
    time_coverage: TimeCoverage | None
    entity_summary: EntitySummary | None
    co_missing: list[list[str]]
    format_issues: list[str]


class Profile(BaseModel):
    dataset: DatasetProfile
    columns: list[ColumnProfile]


# --------------------------------------------------------------------------- helpers


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snake(name: str) -> str:
    return _key(name)


def _try_datetime(s: pd.Series, dayfirst: bool) -> pd.Series | None:
    if s.dtype.kind in "iufb":
        return None
    sample = s.dropna().astype(str)
    if sample.empty:
        return None
    # must look date-like: digits with separators
    looks = sample.str.match(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}").mean()
    if looks < 0.9:
        return None
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
    if parsed.notna().sum() / max(len(sample), 1) >= 0.95:
        return parsed
    return None


def _infer_frequency(dates: pd.Series) -> str | None:
    d = dates.dropna().sort_values().drop_duplicates()
    if len(d) < 3:
        return None
    deltas = d.diff().dropna().dt.days
    mode = deltas.mode()
    if mode.empty:
        return None
    step = int(mode.iloc[0])
    names = {1: "daily", 7: "weekly", 14: "fortnightly", 30: "monthly", 31: "monthly", 365: "yearly", 366: "yearly"}
    return names.get(step, f"every {step} days")


def _kind(s: pd.Series, name: str, rows: int, parsed_dt: pd.Series | None) -> Kind:
    nonnull = s.dropna()
    distinct = nonnull.nunique()
    if parsed_dt is not None:
        return "datetime"
    if s.dtype.kind in "iu" and distinct == rows and rows > 1:
        vals = np.sort(nonnull.to_numpy())
        if np.array_equal(vals, np.arange(vals[0], vals[0] + rows)) or re.search(r"(^|_)(id|instant|index)$", _snake(name)):
            return "id"
    if s.dtype.kind == "b":
        return "boolean"
    if distinct <= 2 and nonnull.astype(str).str.strip().str.lower().isin(BOOL_TOKENS).all():
        return "boolean"
    if s.dtype.kind in "iuf":
        if s.dtype.kind in "iu" and distinct <= 12 and distinct / max(rows, 1) < 0.05:
            return "categorical"
        return "numeric"
    coerced = pd.to_numeric(nonnull, errors="coerce")
    if coerced.notna().mean() >= 0.98 and distinct > 12:
        return "numeric"
    if s.dtype.kind == "O" and re.search(r"(^|_)id$", _snake(name)) and distinct == rows:
        return "id"
    avg_len = nonnull.astype(str).str.len().mean() if not nonnull.empty else 0
    if distinct / max(rows, 1) > 0.5 and avg_len > 20:
        return "text"
    return "categorical"


def _sentinels(s: pd.Series, kind: Kind) -> list[Sentinel]:
    out: list[Sentinel] = []
    nonnull = s.dropna()
    if nonnull.empty:
        return out
    if kind in {"numeric", "id"}:
        num = pd.to_numeric(nonnull, errors="coerce").dropna()
        counts = num.value_counts()
        lo, hi = num.min(), num.max()
        # a placeholder number only counts if it sits at an edge of the range
        for v in PLACEHOLDER_NUMBERS:
            if v in counts.index and v in (lo, hi):
                out.append(Sentinel(value=str(v), count=int(counts[v]), reason="placeholder number at the edge of the range"))
        # a repeated negative value in a column that is otherwise non-negative
        if lo < 0 and num.quantile(0.01) >= 0:
            neg = num[num < 0]
            for v, c in neg.value_counts().head(3).items():
                if c >= 2 and str(v) not in {o.value for o in out}:
                    out.append(Sentinel(value=str(v), count=int(c), reason="negative among non-negative values"))
        # repeated extreme far outside the bulk
        if len(num) >= 50:
            p99, p01, p50 = num.quantile(0.99), num.quantile(0.01), num.quantile(0.5)
            spread = max(p99 - p50, p50 - p01, 1e-9)
            for v, c in ((hi, counts.get(hi, 0)), (lo, counts.get(lo, 0))):
                if c >= max(3, 0.005 * len(num)) and abs(v - p50) > 3 * spread and str(v) not in {o.value for o in out}:
                    out.append(Sentinel(value=str(v), count=int(c), reason="repeated extreme far from the bulk"))
    else:
        strs = nonnull.astype(str)
        low = strs.str.strip().str.lower()
        counts = low.value_counts()
        n = len(low)
        # placeholder text is a minority value; a token that is a quarter of the column is a category
        for tok in PLACEHOLDER_STRINGS:
            if tok in counts.index and counts[tok] / n <= 0.25:
                out.append(Sentinel(value=repr(tok), count=int(counts[tok]), reason="placeholder text"))
    return out


def _format_issues(s: pd.Series, name: str) -> list[str]:
    issues: list[str] = []
    if name != name.strip():
        issues.append("header has leading or trailing whitespace")
    if s.dtype.kind == "O":
        strs = s.dropna().astype(str)
        if not strs.empty:
            if (strs != strs.str.strip()).any():
                issues.append("values have leading or trailing whitespace")
            numeric_share = pd.to_numeric(strs, errors="coerce").notna().mean()
            if 0.05 < numeric_share < 0.95:
                issues.append("mixed numeric and non-numeric values")
    return issues


def _varies_over(
    s: pd.Series, df: pd.DataFrame, entity_cols: list[str], time_col: str | None
) -> VariesOver:
    if not entity_cols and time_col is None:
        return "unknown"
    if s.nunique(dropna=True) <= 1:
        return "neither"
    across_entity = False
    within_entity = False
    if entity_cols:
        per_entity = df.groupby(entity_cols, dropna=False)[s.name].nunique(dropna=True)
        within_entity = bool((per_entity > 1).any())
        # entities differ if their sets of values differ, not just their first values
        value_sets = df.groupby(entity_cols, dropna=False)[s.name].agg(lambda x: frozenset(x.dropna().astype(str)))
        across_entity = value_sets.nunique() > 1
    if time_col is not None and not entity_cols:
        # single entity: anything that changes is changing over time
        return "time"
    if within_entity and across_entity:
        return "both"
    if within_entity:
        return "time"
    if across_entity:
        return "entity"
    return "neither"


def _switch_profile(
    s: pd.Series, df: pd.DataFrame, entity_cols: list[str], time_parsed: pd.Series | None, period: pd.Series | None = None
) -> SwitchProfile | None:
    if not entity_cols:
        return None
    on = s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})
    if time_parsed is not None:
        t = time_parsed
    elif period is not None:
        t = period
    else:
        t = pd.Series(np.arange(len(df)), index=df.index)
    work = pd.DataFrame({"on": on, "t": t})
    for c in entity_cols:
        work[c] = df[c].to_numpy()
    work = work.sort_values(entity_cols + ["t"])
    g = work.groupby(entity_cols, dropna=False)["on"]
    switches = g.apply(lambda x: int((x != x.shift()).sum() - 1))
    any_on = g.any()
    all_on = g.all()
    first_on = work.loc[work["on"], "t"].min() if work["on"].any() else None
    last_on_first = (
        work[work["on"]].groupby(entity_cols, dropna=False)["t"].min().max() if work["on"].any() else None
    )
    fmt = lambda v: (str(pd.Timestamp(v).date()) if time_parsed is not None and v is not None else (str(v) if v is not None else None))  # noqa: E731
    return SwitchProfile(
        entities_that_switch=int((switches > 0).sum()),
        entities_never_on=int((~any_on).sum()),
        entities_always_on=int(all_on.sum()),
        switches_per_entity_median=float(switches.median()),
        first_on=fmt(first_on),
        last_entity_first_on=fmt(last_on_first),
    )


def _candidate_keys(df: pd.DataFrame, kinds: dict[str, Kind], max_pair_cols: int = 12) -> list[list[str]]:
    """Columns that identify a row. Measures (numeric) are never keys, even if unique."""
    rows = len(df)
    keys: list[list[str]] = []
    eligible = [c for c in df.columns if kinds[c] in {"id", "categorical", "datetime", "boolean"}]
    singles = [c for c in eligible if df[c].nunique(dropna=False) == rows]
    keys.extend([[c] for c in singles])
    # pairs among eligible columns that are not already keys and have some cardinality
    cands = [c for c in eligible if c not in singles and df[c].nunique(dropna=False) > 1]
    cands = sorted(cands, key=lambda c: -df[c].nunique(dropna=False))[:max_pair_cols]
    order = {c: i for i, c in enumerate(df.columns)}
    for a, b in combinations(cands, 2):
        if df.groupby([a, b], dropna=False).ngroups == rows:
            keys.append(sorted([a, b], key=order.get))
    return keys


def _co_missing(df: pd.DataFrame) -> list[list[str]]:
    nulls = df.isna()
    cols = [c for c in df.columns if nulls[c].any()]
    out: list[list[str]] = []
    for a, b in combinations(cols, 2):
        if nulls[a].equals(nulls[b]):
            out.append([a, b])
    return out


# --------------------------------------------------------------------------- main


def profile(
    csv_path: str | Path,
    entity_columns: list[str] | None = None,
    time_column: str | None = None,
    dayfirst: bool = False,
) -> Profile:
    path = Path(csv_path)
    df = pd.read_csv(path)
    rows = len(df)
    entity_cols = entity_columns or []
    dataset_issues: list[str] = []

    # Resolve entity/time columns against raw headers, tolerating whitespace in headers.
    header_map = {c.strip(): c for c in df.columns}
    entity_cols = [header_map.get(c, c) for c in entity_cols]
    if time_column is not None:
        time_column = header_map.get(time_column, time_column)
    for c in entity_cols + ([time_column] if time_column else []):
        if c not in df.columns:
            raise ValueError(f"column not in file: {c!r}")
    if any(c != c.strip() for c in df.columns):
        dataset_issues.append("one or more headers have leading or trailing whitespace")

    parsed: dict[str, pd.Series | None] = {c: _try_datetime(df[c], dayfirst) for c in df.columns}
    time_parsed = parsed[time_column] if time_column else None
    time_is_period = False  # integer periods such as year=63..92 or week=1..4
    if time_column and time_parsed is None:
        if df[time_column].dtype.kind in "iu":
            time_is_period = True
            time_parsed = df[time_column]
        else:
            dataset_issues.append(f"time column {time_column!r} is neither a date nor an integer period")

    columns: list[ColumnProfile] = []
    for c in df.columns:
        s = df[c]
        nonnull = s.dropna()
        kind: Kind = "id" if c in entity_cols else _kind(s, c, rows, parsed[c])
        distinct = int(nonnull.nunique())
        cp = ColumnProfile(
            name=c,
            key=_snake(c),
            kind=kind,
            nulls=int(s.isna().sum()),
            null_rate=float(round(s.isna().mean(), 6)),
            distinct=distinct,
            constant=distinct <= 1,
            observed_sentinels=_sentinels(s, kind),
            format_issues=_format_issues(s, c),
        )
        if kind in {"numeric", "id"}:
            num = pd.to_numeric(nonnull, errors="coerce").dropna()
            if not num.empty:
                q = num.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
                cp.numeric = NumericStats(
                    min=float(num.min()), max=float(num.max()), mean=float(round(num.mean(), 6)),
                    p01=float(q[0.01]), p25=float(q[0.25]), p50=float(q[0.5]), p75=float(q[0.75]), p99=float(q[0.99]),
                )
        if kind in {"categorical", "boolean"}:
            vc = nonnull.astype(str).value_counts().head(8)
            cp.top_values = [TopValue(value=str(v), count=int(n), share=float(round(n / max(rows, 1), 4))) for v, n in vc.items()]
        if kind == "datetime" and parsed[c] is not None:
            d = parsed[c].dropna()
            cp.datetime = DatetimeStats(first=str(d.min().date()), last=str(d.max().date()), inferred_frequency=_infer_frequency(d))
        cp.varies_over = _varies_over(s, df, entity_cols, time_column)
        if kind == "boolean" and cp.varies_over in {"time", "both"} and entity_cols:
            cp.switch = _switch_profile(s, df, entity_cols, None if time_is_period else time_parsed, period=time_parsed if time_is_period else None)
        columns.append(cp)

    time_cov = None
    if time_column and time_parsed is not None and time_is_period:
        uniq = np.sort(time_parsed.dropna().unique())
        step = int(pd.Series(np.diff(uniq)).mode().iloc[0]) if len(uniq) > 1 else 1
        expected = (int(uniq.max()) - int(uniq.min())) // step + 1
        time_cov = TimeCoverage(column=time_column, first=str(int(uniq.min())), last=str(int(uniq.max())),
                                inferred_frequency=f"integer period, step {step}", gaps=int(expected - len(uniq)))
    elif time_column and time_parsed is not None:
        d = time_parsed.dropna()
        freq = _infer_frequency(d)
        gaps = 0
        if freq in {"daily", "weekly"}:
            step = 1 if freq == "daily" else 7
            uniq = d.sort_values().drop_duplicates()
            expected = (uniq.max() - uniq.min()).days // step + 1
            gaps = int(expected - len(uniq))
        time_cov = TimeCoverage(column=time_column, first=str(d.min().date()), last=str(d.max().date()), inferred_frequency=freq, gaps=gaps)

    entity_summary = None
    if entity_cols:
        sizes = df.groupby(entity_cols, dropna=False).size()
        entity_summary = EntitySummary(
            columns=entity_cols, entities=int(len(sizes)),
            rows_per_entity_min=int(sizes.min()), rows_per_entity_median=float(sizes.median()), rows_per_entity_max=int(sizes.max()),
        )

    keys = _candidate_keys(df, {cp.name: cp.kind for cp in columns})
    grain = min(keys, key=len) if keys else None

    dataset = DatasetProfile(
        profiler_version=PROFILER_VERSION,
        file=str(path),
        file_hash=_sha256(path),
        rows=rows,
        columns=len(df.columns),
        duplicate_rows=int(df.duplicated().sum()),
        candidate_keys=keys,
        grain=grain,
        time_coverage=time_cov,
        entity_summary=entity_summary,
        co_missing=_co_missing(df),
        format_issues=dataset_issues,
    )
    return Profile(dataset=dataset, columns=columns)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Profile a CSV deterministically.")
    ap.add_argument("csv")
    ap.add_argument("--entity", action="append", default=[], help="entity column; repeatable")
    ap.add_argument("--time", default=None, help="time column")
    ap.add_argument("--dayfirst", action="store_true", help="parse dates day-first")
    ap.add_argument("-o", "--out", default=None, help="write JSON here instead of stdout")
    args = ap.parse_args(argv)
    p = profile(args.csv, entity_columns=args.entity or None, time_column=args.time, dayfirst=args.dayfirst)
    text = json.dumps(p.model_dump(), indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
