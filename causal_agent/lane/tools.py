"""The read-only data tools a lane may offer a reasoning episode. Each takes column names, runs on the intake table and returns
a short sentence with a number where there is one; the episode gives the result an address the record may cite.

One rule, kept here and not in a prompt: no tool joins the outcome with the treatment before the design is frozen. Choosing
roles or modifiers by peeking at the effect is the one thing this structure must make impossible. `by_arm(outcome)`,
`association(outcome, treatment)`, `redundancy(outcome, treatment)` and `cells([.., outcome])` are refused with a reason the
model reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import pandas as pd
from langchain_core.tools import StructuredTool

from causal_agent.common.addresses import key as _key
from causal_agent.memory import facts as FX

MAX_CELL_COLUMNS = 3
MAX_LEVELS = 6
MAX_CELL_LINES = 12
NAMES = ("describe", "by_arm", "association", "redundancy", "cells", "timing")


class Refused(Exception):
    """The tool did not run; the message is what the model reads."""


class Result(NamedTuple):
    text: str
    value: float | None = None


def _levels(s: pd.Series) -> pd.Series:
    """A column as a small set of levels: its values when few, quantile bins when numeric and many."""
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > MAX_LEVELS:
        return pd.qcut(s, q=MAX_LEVELS, duplicates="drop").astype(str)
    return s.astype(str)


@dataclass
class Tools:
    """The tools over one table. `outcome` and `treatment` are raw column names; `treated` is the mask of the treated rows when
    the arms are settled; `timing_of` maps a raw name to before, at, after or unknown; `frozen` lifts the outcome rule after the
    design is frozen."""

    df: pd.DataFrame
    outcome: str | None = None
    treatment: str | None = None
    treated: pd.Series | None = None
    timing_of: dict[str, str] | None = None
    frozen: bool = False

    # ------------------------------------------------------------------ resolving and the rule
    def column(self, name: str) -> str:
        """The raw column a name or a key points at."""
        if name in self.df.columns:
            return name
        for c in self.df.columns:
            if _key(str(c)) == _key(name):
                return str(c)
        raise Refused(f"refused: {name!r} is not a column of the table")

    def _guard(self, cols: list[str], *, by_arm: bool) -> None:
        if self.frozen or self.outcome is None or self.outcome not in cols:
            return
        if by_arm or self.treatment in cols:
            raise Refused(
                f"refused: that would join the outcome {self.outcome!r} with the treatment before the design is frozen; "
                "the effect is the run's to find, not a fact to choose the design by"
            )

    def _arms(self) -> pd.Series:
        if self.treated is None:
            raise Refused("refused: which rows are treated is not settled, so there are no arms")
        return self.treated

    # ------------------------------------------------------------------ the tools
    def describe(self, column: str) -> Result:
        """What one column holds: its kind, the rows with a value, the distinct values, and its top levels or its quartiles."""
        c = self.column(column)
        s = self.df[c]
        n, nulls, distinct = int(s.notna().sum()), int(s.isna().sum()), int(s.nunique(dropna=True))
        head = f"{c!r}: {n} rows with a value, {nulls} missing, {distinct} distinct"
        if pd.api.types.is_numeric_dtype(s) and distinct > MAX_LEVELS:
            x = pd.to_numeric(s, errors="coerce").dropna()
            q = x.quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
            return Result(
                f"{head}; a number: min {q[0]:.3g}, quartiles {q[1]:.3g} / {q[2]:.3g} / {q[3]:.3g}, max {q[4]:.3g}, mean {x.mean():.3g}", float(x.mean())
            )
        vc = s.astype(str).value_counts(normalize=True)
        top = "; ".join(f"{v}: {p:.0%}" for v, p in vc.head(MAX_LEVELS).items())
        more = f"; {distinct - MAX_LEVELS} more levels" if distinct > MAX_LEVELS else ""
        return Result(f"{head}; levels: {top}{more}", float(distinct))

    def by_arm(self, column: str) -> Result:
        """One column in each arm: the means and the standardised difference for a number, the share of each level for a category."""
        c = self.column(column)
        treated = self._arms()
        self._guard([c], by_arm=True)
        text, v = FX.by_arm(self.df[c], treated, c)
        return Result(text, v)

    def association(self, a: str, b: str) -> Result:
        """How two columns move together: Pearson's r for two numbers, Cramér's V for two categories, the correlation ratio for one of each."""
        x, y = self.column(a), self.column(b)
        if x == y:
            raise Refused("refused: the same column twice")
        self._guard([x, y], by_arm=False)
        v, how = FX.association(self.df[x], self.df[y])
        if v is None:
            return Result(f"{x!r} and {y!r}: {how} not computable (a constant column or too few rows)")
        return Result(f"{x!r} and {y!r}: {how} {v:.2f}", v)

    def redundancy(self, a: str, b: str) -> Result:
        """Whether two columns carry the same information or one sits inside the other: every value of one maps to one value of the other."""
        x, y = self.column(a), self.column(b)
        if x == y:
            raise Refused("refused: the same column twice")
        self._guard([x, y], by_arm=False)
        r = FX.redundancy(self.df[x], self.df[y])
        if r == "same":
            return Result(f"{x!r} and {y!r} carry the same information: each value of one maps to one value of the other", 1.0)
        if r == "a in b":
            return Result(f"{x!r} sits inside {y!r}: every value of {x!r} maps to one value of {y!r}, and {y!r} has fewer levels", 1.0)
        if r == "b in a":
            return Result(f"{y!r} sits inside {x!r}: every value of {y!r} maps to one value of {x!r}, and {x!r} has fewer levels", 1.0)
        return Result(f"{x!r} and {y!r}: neither determines the other", 0.0)

    def cells(self, columns: list[str]) -> Result:
        """Rows per arm in every cell of these columns, the overlap table; a number with many values is cut into six bins. At most three columns."""
        cols = [self.column(c) for c in columns]
        if not cols:
            raise Refused("refused: name at least one column")
        if len(cols) > MAX_CELL_COLUMNS:
            raise Refused(f"refused: at most {MAX_CELL_COLUMNS} columns in one table")
        treated = self._arms()
        self._guard(cols, by_arm=True)
        lv = pd.concat([_levels(self.df[c]).rename(c) for c in cols], axis=1)
        lv["_arm"] = treated.map({True: "treated", False: "control"})
        t = lv.groupby(cols + ["_arm"], observed=True).size().unstack("_arm", fill_value=0)
        for arm in ("treated", "control"):
            if arm not in t.columns:
                t[arm] = 0
        t = t[["treated", "control"]]
        smallest = int(t.min(axis=1).min()) if len(t) else 0
        empty = int((t.min(axis=1) == 0).sum())
        lines = []
        for idx, row in t.head(MAX_CELL_LINES).iterrows():
            parts = idx if isinstance(idx, tuple) else (idx,)
            lines.append(", ".join(f"{c}={v}" for c, v in zip(cols, parts, strict=True)) + f": treated {int(row['treated'])}, control {int(row['control'])}")
        more = f"; {len(t) - MAX_CELL_LINES} more cells" if len(t) > MAX_CELL_LINES else ""
        head = f"{len(t)} cells over {', '.join(cols)}; smallest arm in a cell {smallest}; {empty} cell{'s' if empty != 1 else ''} with one arm missing"
        return Result(head + ". " + "; ".join(lines) + more, float(smallest))

    def timing(self) -> Result:
        """Every column's place in time against the treatment, as the pack settled it, and which are unknown."""
        when: dict[str, list[str]] = {"before": [], "at": [], "after": [], "unknown": []}
        for c in self.df.columns:
            if c in (self.outcome, self.treatment):
                continue
            w = (self.timing_of or {}).get(str(c), "unknown")
            when[w if w in when else "unknown"].append(str(c))
        return Result("; ".join(f"{k}: {', '.join(v) if v else 'none'}" for k, v in when.items()), float(len(when["unknown"])))

    # ------------------------------------------------------------------ for the episode
    def call(self, name: str, args: dict[str, Any]) -> Result:
        """Run one tool by name; a wrong name or wrong arguments are a refusal like any other."""
        if name not in NAMES:
            raise Refused(f"refused: no tool named {name!r}; the tools are {', '.join(NAMES)}")
        try:
            return getattr(self, name)(**args)
        except TypeError as e:
            raise Refused(f"refused: {name} could not run with {args!r}: {e}") from e

    def schemas(self) -> list[StructuredTool]:
        """The tools as the model sees them, bound to this table."""
        return [StructuredTool.from_function(func=getattr(self, n), name=n, description=str(getattr(Tools, n).__doc__)) for n in NAMES]
