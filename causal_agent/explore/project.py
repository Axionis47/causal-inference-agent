"""The no-join rule for free-form code. The lane's tools refuse a join of the outcome with the treatment (`lane/tools.py`); the
data facts never compute one (`memory/facts.py`); a script may do anything with the table it reads, so before the design has
run the table sent to the agent has one of the pair taken out. Which one is code: the outcome, unless the ask names the outcome
and not the treatment, then the treatment. An ask naming both is refused before any model call."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from causal_agent.common import trace
from causal_agent.common.addresses import key

Drop = Literal["outcome", "treatment", "both"]
ROW = "__row"  # a new version keeps every row's place: the agent sees this column and must keep it on every row it keeps


def mentions(text: str, column: str | None) -> bool:
    """Whether the words name the column: by the file's own name or by its key, with underscores read as spaces."""
    if not column:
        return False
    low = " ".join(text.lower().split())
    k = key(column)
    return column.lower() in low or k in low or k.replace("_", " ") in low


def choose_drop(ask: str, outcome: str | None, treatment: str | None, frozen: bool) -> Drop | None:
    """Which of the pair to take out of the table, by code. None after the freeze or when either column is unknown."""
    if frozen or not outcome or not treatment:
        return None
    o, t = mentions(ask, outcome), mentions(ask, treatment)
    if o and t:
        return "both"
    if o and not t:
        return "treatment"
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
