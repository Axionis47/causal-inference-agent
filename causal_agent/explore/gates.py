"""What code checks of the exploration agent: a script before it runs, and the record before it is taken. The record is
refused with the reason and the episode asks again; the agent never argues its way past a check."""

from __future__ import annotations

import ast
import json
import re
from typing import TYPE_CHECKING

import pandas as pd

from causal_agent.common.numbers import grounded, numbers_in
from causal_agent.explore.contracts import Exploration, ExploreRequest
from causal_agent.explore.project import mentions

if TYPE_CHECKING:
    from causal_agent.explore.tools import CodeTools

RESERVED = ("table", "figure")  # address segments the artifact keeps for its own products
COLUMN_METHODS = {"groupby", "get", "pop", "drop", "dropna", "sort_values", "set_index", "filter", "pivot", "pivot_table", "melt"}
COLUMN_KEYWORDS = {"by", "columns", "subset", "on", "index", "values", "id_vars", "value_vars", "x", "y", "hue"}


def not_a_script(code: str) -> str:
    """Why the code is not a Python script, or empty. Words where the script should be are the usual case."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        head = code.strip().splitlines()[0][:160] if code.strip() else ""
        return f"the code must be only the Python script, no words before or after it (line {e.lineno}: {e.msg}). It began: {head!r}"
    return ""


def names_dropped_column(code: str, name: str | None) -> str:
    """Reject exact column literals or attributes, not substrings of Python names. Projection is the data fence."""
    if not name:
        return ""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return not_a_script(code)

    def literal(node: ast.AST) -> bool:
        return any(isinstance(part, ast.Constant) and isinstance(part.value, str) and part.value == name for part in ast.walk(node))

    def selects(node: ast.AST) -> bool:
        if isinstance(node, ast.Subscript):
            return literal(node.slice)
        if isinstance(node, ast.Attribute):
            return node.attr == name
        if not isinstance(node, ast.Call):
            return False
        if any(keyword.arg in COLUMN_KEYWORDS and literal(keyword.value) for keyword in node.keywords):
            return True
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in COLUMN_METHODS:
                return any(literal(arg) for arg in node.args)
            if node.func.attr in {"query", "eval"}:
                return any(isinstance(arg, ast.Constant) and isinstance(arg.value, str) and mentions(arg.value, name) for arg in node.args)
        if isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) > 1:
            return literal(node.args[1])
        return False

    named = any(selects(node) for node in ast.walk(tree))
    if named:
        return f"the script names {name!r}, which is not in the table you were given: it is held back until the design has run"
    return ""


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def gate_record(record: Exploration, req: ExploreRequest, tools: CodeTools) -> list[str]:
    """Why the record is refused, or nothing."""
    errors: list[str] = []
    step = tools.step(record.step)
    if step is None:
        return [
            f"step {record.step} did not run; the steps so far are 1 to {len(tools.steps)}"
            if tools.steps
            else "no script has run yet; run one with run_code first"
        ]
    if not step.ok:
        return [f"step {record.step} failed; name a step that ran, or run another"]
    folder = tools.out_dir / f"step_{record.step}"
    bad = [n for n in record.facts if n in RESERVED]
    if bad:
        errors.append(f"a fact may not be named {' or '.join(repr(b) for b in bad)}; those names are kept for the table and the picture")
    if record.facts:
        if step.facts is None:
            errors.append(f"step {record.step} wrote no facts.json that is an object of numbers; the facts you stand behind must be in it")
        else:
            missing = [n for n in record.facts if n not in step.facts]
            if missing:
                errors.append(f"step {record.step}'s facts.json lacks: {', '.join(missing)}")
    if record.figure and not ((folder / "figure.png").is_file() and (folder / "figure.png").stat().st_size > 0):
        errors.append(f"figure is true but step {record.step} wrote no figure.png")
    if record.table:
        t = folder / record.table
        if record.table not in step.files or not t.is_file():
            errors.append(f"table names {record.table!r}, which step {record.step} did not write")
        elif not _parses(t):
            errors.append(f"table {record.table!r} is not a readable, non-empty CSV or JSON file")
    if not (record.facts or record.figure or record.table):
        errors.append("the record names no product: at least one fact, a figure, or a table")
    if req.product == "table":
        errors += new_table_errors(record, req, folder)
    if req.outcome and req.treatment and not req.frozen:
        joined = [s for s in _sentences(record.summary) if mentions(s, req.outcome) and mentions(s, req.treatment)]
        joined += [n for n in record.facts if mentions(n, req.outcome) and mentions(n, req.treatment)]
        if joined:
            errors.append("before the design has run nothing may relate the outcome to the treatment; this does: " + "; ".join(repr(j) for j in joined))
    known = list((step.facts or {}).values()) + numbers_in(req.context) + numbers_in(req.ask)  # the ask's own numbers were told too
    loose = [v for v in numbers_in(record.summary) if not grounded(v, known)]
    if loose:
        errors.append("the summary states numbers that are in neither facts.json nor what you were told: " + ", ".join(f"{v:g}" for v in loose))
    return errors


def _parses(path) -> bool:
    try:
        if path.suffix == ".csv":
            return len(pd.read_csv(path)) > 0
        if path.suffix == ".json":
            doc = json.loads(path.read_text())
            return bool(doc)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return False


def new_table_errors(record: Exploration, req: ExploreRequest, folder) -> list[str]:
    """What a new version of the table is held to: the step wrote table.csv; every row carries the row id it was given, once, and
    no row is new; the columns the pair needs are still there."""
    from causal_agent.explore.project import ROW

    if record.table != "table.csv":
        return ['a new version of the table is table.csv, written by the step named; set table to "table.csv"']
    path = folder / "table.csv"
    if not path.is_file():
        return [f"step {record.step} wrote no table.csv"]
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001  (whatever pandas raises is the reason)
        return [f"table.csv does not read as a CSV: {e}"]
    errors: list[str] = []
    if ROW not in df.columns:
        return [f"table.csv has no {ROW} column; keep it on every row exactly as given"]
    ids = df[ROW]
    if ids.isna().any() or not pd.api.types.is_integer_dtype(ids):
        errors.append(f"{ROW} must be the integer given on every row; a row without one is a row that was not in the table")
    else:
        if ids.duplicated().any():
            errors.append(f"{ROW} repeats; a row is kept once or dropped, never doubled")
        if req.rows and ((ids < 0) | (ids >= req.rows)).any():
            errors.append(f"{ROW} holds values outside 0 to {req.rows - 1}: a row that was not in the table cannot be added")
    gone = [c for c in req.keep if c not in df.columns]
    if gone:
        errors.append("the new table must keep " + ", ".join(repr(c) for c in req.keep) + "; missing: " + ", ".join(repr(c) for c in gone))
    if len(df) == 0:
        errors.append("the new table has no rows")
    return errors
