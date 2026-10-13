"""The outcome fence for free-form code. Before a frozen run, every executable request gets a table without the known
outcome. Assignment can be reconstructed from other columns, so removing only the treatment cannot protect the reasoning
boundary. An ask naming both is refused before any model call. Deterministic profiles own univariate outcome summaries."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd

from causal_agent.common import trace
from causal_agent.common.addresses import key

Drop = Literal["outcome", "both"]
ROW = "__row"  # a new version keeps every row's place: the agent sees this column and must keep it on every row it keeps


def mentions(text: str, column: str | None) -> bool:
    """Whether the words name the column: by the file's own name or by its key, with underscores read as spaces."""
    if not column:
        return False
    low = " ".join(text.lower().replace("_", " ").split())
    forms = {" ".join(value.lower().replace("_", " ").split()) for value in (column, key(column))}
    return any(form and re.search(r"(?<!\w)" + re.escape(form) + r"(?!\w)", low) for form in forms)


def choose_drop(ask: str, outcome: str | None, treatment: str | None, frozen: bool) -> Drop | None:
    """Withhold the outcome before the freeze, even for an outcome-only ask or when treatment is unknown."""
    if frozen or not outcome:
        return None
    if mentions(ask, outcome) and mentions(ask, treatment):
        return "both"
    return "outcome"


def project(csv: Path, out_dir: Path, *, drop: str | None, row_id: bool = False) -> Path:
    """The table as the sandbox reads it, written under the folder: without one column when one is held back, and with the
    row id first when a new version is asked for. One tool span naming the column dropped and what was kept; the file is
    derivable from the dataset, so it is not attached."""
    with trace.span("project", run_type="tool", inputs={"csv": str(csv), "dropped": drop, "row_id": row_id}) as s:
        df = pd.read_csv(csv)
        if drop is not None and drop not in df.columns:
            raise KeyError(f"the column to drop, {drop!r}, is not in {csv}")
        kept = df.drop(columns=[drop]) if drop is not None else df
        if row_id:
            kept = kept.copy()
            kept.insert(0, ROW, range(len(kept)))
        out = out_dir / "input"
        out.mkdir(parents=True, exist_ok=True)
        path = out / "table.csv"
        kept.to_csv(path, index=False)
        s.end(outputs={"path": str(path), "rows": int(len(kept)), "columns": list(kept.columns)})
        return path
