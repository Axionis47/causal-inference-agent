"""The table a lane works on, by code: the CSV the pack names, the columns the pack names (the frame's and the design
block's), the scope's row filter and time window applied, and a record of everything that could not be applied.

    intake = load(h, "did")             # or raises IntakeStop with a Feasibility the lane returns as its stop
    intake.table, intake.columns, intake.declines, intake.facts

The filter grammar is small on purpose: `col == v`, `col != v`, `col >= v`, `col <= v`, `col > v`, `col < v`,
`col in [a, b]`, joined by `and`. The window grammar: `from A to B`, `A..B`, `A to B`, `>= A`, `<= A`, `after A`,
`before A`, `since A`, `until A`. A filter or window the code cannot read is a Decline, never a guess."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from causal_agent.common import config
from causal_agent.common.addresses import key as _key
from causal_agent.common.contracts import AdjustmentDesign, Decline, DidDesign, Feasibility, Handoff, RdDesign
from causal_agent.profile import datasets as DS


class IntakeStop(Exception):
    """The lane cannot start: the outcome or the treatment is not in the file, or the file is not there."""

    def __init__(self, feasibility: Feasibility):
        super().__init__(feasibility.reason)
        self.feasibility = feasibility


@dataclass
class Intake:
    table: pd.DataFrame
    columns: dict[str, str]  # key -> raw name, for every column loaded
    run_dir: Path
    table_path: Path
    declines: list[Decline] = field(default_factory=list)
    facts: dict = field(default_factory=dict)  # check_facts["intake"]


# ------------------------------------------------------------------ what the pack names


def design_columns(h: Handoff) -> list[str]:
    """Every column the family block names, by key. A column the desk put in the block is loaded even when the frame forgot it."""
    b = h.design
    names: list = []
    if isinstance(b, AdjustmentDesign):
        names += [b.instrument, b.mediator] + list(b.adjustment_candidates)
    elif isinstance(b, DidDesign):
        names += [b.unit, b.time, (b.treated_group or {}).get("column"), b.cluster_level] + list(b.controls_allowed)
    elif isinstance(b, RdDesign):
        names += [b.score, (b.takeup or {}).get("column"), b.cluster] + list(b.covariates_allowed)
    return [_key(n) for n in names if n]


def wanted_columns(h: Handoff, extra: list[str] | None = None) -> list[str]:
    t, y = (_key(h.treatment) if h.treatment else None), _key(h.outcome)
    rel = [_key(c.column) for c in h.relevant_columns]
    return [k for k in dict.fromkeys([t, y] + rel + design_columns(h) + [_key(x) for x in (extra or []) if x]) if k]


# ------------------------------------------------------------------ the filter


_OPS = ("==", "!=", ">=", "<=", ">", "<", "=", " in ")
_CLAUSE = re.compile(r"^\s*(?P<col>.+?)\s*(?P<op>==|!=|>=|<=|>|<|=|\bin\b)\s*(?P<val>.+?)\s*$", re.IGNORECASE)


def _unquote(s: str) -> str:
    s = s.strip()
    return s[1:-1] if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"" else s


def _values(s: str) -> list[str]:
    s = s.strip()
    if s[:1] in "[(" and s[-1:] in "])":
        s = s[1:-1]
    return [_unquote(v) for v in s.split(",") if v.strip()]


def _typed(series: pd.Series, v: str):
    if pd.api.types.is_numeric_dtype(series):
        return float(v)
    return str(v)


def parse_filter(text: str) -> list[tuple[str, str, list[str]]] | None:
    """Clauses (column, op, values) or None when the text is not in the grammar."""
    out = []
    for clause in re.split(r"\s+and\s+", text.strip(), flags=re.IGNORECASE):
        m = _CLAUSE.match(clause)
        if not m:
            return None
        op = m.group("op").lower()
        op = "==" if op == "=" else op
        vals = _values(m.group("val")) if op == "in" else [_unquote(m.group("val"))]
        if not vals:
            return None
        out.append((m.group("col").strip(), op, vals))
    return out or None


_BLANK = re.compile(r"^\s*(null|none|n/?a|no filter|no window|whole|all( \w+){0,3}|every \w+( \w+){0,2})\s*\.?\s*$", re.IGNORECASE)


def blank(text: str | None) -> bool:
    """A scope the frame wrote as words for 'nothing': the word null, none, all students, every row. Not a decline; there is nothing to apply."""
    return not text or bool(_BLANK.match(str(text)))


def apply_filter(table: pd.DataFrame, text: str | None, stage: str = "load") -> tuple[pd.DataFrame, Decline | None, dict]:
    """The rows the filter keeps, or the table unchanged with a Decline saying why."""
    if blank(text):
        return table, None, {}
    clauses = parse_filter(text or "")
    if clauses is None:
        return (
            table,
            Decline(
                stage=stage,
                kind="declined",
                about="scope.population_filter",
                pack_value=text,
                check="intake.filter_unparsed",
                reason="the filter is not in a form the code can apply (col == v, !=, >=, <=, >, <, in [a, b], joined by and); every row was kept",
            ),
            {},
        )
    mask = pd.Series(True, index=table.index)
    for col, op, vals in clauses:
        k = _key(col)
        if k not in table.columns:
            return (
                table,
                Decline(
                    stage=stage,
                    kind="declined",
                    about="scope.population_filter",
                    pack_value=text,
                    check="intake.filter_unknown_column",
                    reason=f"the filter names {col!r}, which is not a column in the table; every row was kept",
                ),
                {},
            )
        s = table[k]
        try:
            typed = [_typed(s, v) for v in vals]
        except ValueError:
            return (
                table,
                Decline(
                    stage=stage,
                    kind="declined",
                    about="scope.population_filter",
                    pack_value=text,
                    check="intake.filter_value_type",
                    reason=f"the filter compares {col!r} with {vals}, which is not a number like the column; every row was kept",
                ),
                {},
            )
        left = s if pd.api.types.is_numeric_dtype(s) else s.astype(str)
        if op == "in":
            m = left.isin(typed)
        else:
            v = typed[0]
            m = {"==": left == v, "!=": left != v, ">=": left >= v, "<=": left <= v, ">": left > v, "<": left < v}[op]
        mask &= m.fillna(False)
    kept = table[mask]
    return kept, None, {"filter": text, "rows_before": int(len(table)), "rows_after": int(len(kept))}


# ------------------------------------------------------------------ the window


_WINDOW = [
    (re.compile(r"^\s*from\s+(.+?)\s+(?:to|until|through)\s+(.+?)\s*$", re.I), ("ge", "le")),
    (re.compile(r"^\s*(.+?)\s*\.\.\s*(.+?)\s*$"), ("ge", "le")),
    (re.compile(r"^\s*(.+?)\s+to\s+(.+?)\s*$", re.I), ("ge", "le")),
    (re.compile(r"^\s*(?:>=|since|from)\s*(.+?)\s*$", re.I), ("ge", None)),
    (re.compile(r"^\s*(?:<=|until|through)\s*(.+?)\s*$", re.I), (None, "le")),
    (re.compile(r"^\s*(?:>|after)\s*(.+?)\s*$", re.I), ("gt", None)),
    (re.compile(r"^\s*(?:<|before)\s*(.+?)\s*$", re.I), (None, "lt")),
]


def parse_window(text: str) -> tuple[str | None, str | None, str | None, str | None] | None:
    """(low, low_op, high, high_op) as text, or None when the window is not in the grammar."""
    for rx, (lo_op, hi_op) in _WINDOW:
        m = rx.match(text)
        if m:
            g = list(m.groups())
            if lo_op and hi_op:
                return g[0], lo_op, g[1], hi_op
            if lo_op:
                return g[0], lo_op, None, None
            return None, None, g[0], hi_op
    return None


def _parse_like(time: pd.Series, value: str):
    if pd.api.types.is_numeric_dtype(time):
        return float(value)
    return pd.to_datetime(value)


def apply_window(table: pd.DataFrame, text: str | None, time_key: str | None, stage: str = "load") -> tuple[pd.DataFrame, Decline | None, dict]:
    """The rows inside the window on the time column, or the table unchanged with a Decline."""
    if blank(text):
        return table, None, {}
    if not time_key or time_key not in table.columns:
        return (
            table,
            Decline(
                stage=stage,
                kind="declined",
                about="scope.window",
                pack_value=text,
                check="intake.window_no_time_column",
                reason="the question names a time window but the pack names no time column in the table; every row was kept",
            ),
            {},
        )
    parsed = parse_window(text or "")
    if parsed is None:
        return (
            table,
            Decline(
                stage=stage,
                kind="declined",
                about="scope.window",
                pack_value=text,
                check="intake.window_unparsed",
                reason="the window is not in a form the code can apply (from A to B, A..B, >= A, <= A, after A, before A); every row was kept",
            ),
            {},
        )
    lo, lo_op, hi, hi_op = parsed
    t = table[time_key]
    if not pd.api.types.is_numeric_dtype(t):
        try:
            t = pd.to_datetime(t)
        except Exception:
            return (
                table,
                Decline(
                    stage=stage,
                    kind="declined",
                    about="scope.window",
                    pack_value=text,
                    check="intake.window_time_unparsed",
                    reason=f"{time_key!r} is neither numeric nor a date the code can read; every row was kept",
                ),
                {},
            )
    try:
        mask = pd.Series(True, index=table.index)
        if lo is not None:
            v = _parse_like(t, lo)
            mask &= (t >= v) if lo_op == "ge" else (t > v)
        if hi is not None:
            v = _parse_like(t, hi)
            mask &= (t <= v) if hi_op == "le" else (t < v)
    except Exception:
        return (
            table,
            Decline(
                stage=stage,
                kind="declined",
                about="scope.window",
                pack_value=text,
                check="intake.window_value_type",
                reason=f"the window's bounds are not like the values of {time_key!r}; every row was kept",
            ),
            {},
        )
    kept = table[mask.fillna(False)]
    return kept, None, {"window": text, "time_column": time_key, "rows_before": int(len(table)), "rows_after": int(len(kept))}


# ------------------------------------------------------------------ load


def time_key(h: Handoff, entry: dict | None = None) -> str | None:
    """The time column the pack names: the panel block's, the change's date column, or the dataset entry's."""
    b = h.design
    if isinstance(b, DidDesign) and b.time:
        return _key(b.time)
    if h.change.get("date_column"):
        return _key(h.change["date_column"])
    e = entry or {}
    return _key(e["time"]) if e.get("time") else None


def load(h: Handoff, tag: str, *, extra: list[str] | None = None, dropna: bool = True, stage: str = "load") -> Intake:
    """The table for one run. Raises IntakeStop when the outcome or the treatment is missing; every other trouble is a Decline."""
    entry = DS.dataset_entries().get(h.pack_name) or {}  # through the module, so a test can point it at its own index
    csv = h.csv or entry.get("csv")
    if not csv:
        raise IntakeStop(
            Feasibility(stage=stage, reason="the hand-off names no file", facts=[f"pack {h.pack_name!r}"], what_would_fix="a hand-off with a csv path")
        )
    raw = pd.read_csv(Path(DS.ROOT) / csv)
    columns = {_key(c): c for c in raw.columns}
    raw.columns = [_key(c) for c in raw.columns]
    t, y = (_key(h.treatment) if h.treatment else None), _key(h.outcome)
    declines: list[Decline] = []
    if y not in raw.columns or (t and t not in raw.columns):
        missing = [k for k in (t, y) if k and k not in raw.columns]
        raise IntakeStop(
            Feasibility(
                stage=stage,
                reason="a column the hand-off names is not in the file",
                facts=[f"missing: {missing}"],
                what_would_fix="a hand-off whose columns exist in the file",
            )
        )
    wanted = wanted_columns(h, extra)
    for k in [k for k in wanted if k not in raw.columns]:
        declines.append(
            Decline(
                stage=stage,
                kind="declined",
                about=f"col:{k}",
                check="intake.column_missing",
                reason="the pack names it and the file has no such column; it was not loaded",
            )
        )
    wanted = [k for k in wanted if k in raw.columns]
    table = raw[wanted]
    facts: dict = {}
    table, d, f = apply_filter(table, h.scope.population_filter, stage)
    if d:
        declines.append(d)
    facts.update(f)
    table, d, f = apply_window(table, h.scope.window, time_key(h, entry), stage)
    if d:
        declines.append(d)
    facts.update(f)
    if dropna:
        table = table.dropna()
    run_dir = config.get().paths.runs / f"{h.pack_name}-{tag}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    table_path = run_dir / "table.csv"
    table.to_csv(table_path, index=False)
    return Intake(table=table, columns={k: columns[k] for k in wanted}, run_dir=run_dir, table_path=table_path, declines=declines, facts=facts)
