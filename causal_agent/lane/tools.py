"""The read-only data tools a lane may offer a reasoning episode. Each takes column names, runs on the lane's table and returns
a short sentence with a number where there is one; the episode gives the result an address the record may cite.

One rule, kept here and not in a prompt: no tool joins the outcome with the treatment before the design is frozen. Choosing
roles or modifiers by peeking at the effect is the one thing this structure must make impossible. A lane whose design has rows
on which the outcome is evidence and not the effect (a panel's rows before the change) names them in `allow_outcome_rows`, and a
tool that would join the pair runs on those rows only and says so; every other lane leaves the mask empty and the join is refused.

Six tools run on any table. Four more run only on a table with the shape they need and are offered only then: two on a panel
(`unit`, `time`, `treated`), two on a table with a recentred score (`x`, the cutoff at zero and the treated side positive).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from langchain_core.tools import StructuredTool

from causal_agent.common.addresses import key as _key
from causal_agent.memory import facts as FX

MAX_CELL_COLUMNS = 3
MAX_LEVELS = 6
MAX_CELL_LINES = 12
MAX_PERIODS = 12
MAX_BINS = 20
NAMES = ("describe", "by_arm", "association", "redundancy", "cells", "timing", "by_group_over_time", "composition", "by_side_near", "score_histogram")
PANEL = ("unit", "time", "treated")
SCORE = ("x",)
NEEDS: dict[str, tuple[str, ...]] = {"by_group_over_time": PANEL, "composition": PANEL, "by_side_near": SCORE, "score_histogram": SCORE}


class Refused(Exception):
    """The tool did not run; the message is what the model reads."""


class Result(NamedTuple):
    text: str
    value: float | None = None


def _period(v: Any) -> str:
    """A period as the reader knows it: a whole number without a trailing .0, anything else as written."""
    if isinstance(v, (float, np.floating)) and np.isfinite(v) and float(v) == int(v):
        return str(int(v))
    return str(v)


def _levels(s: pd.Series) -> pd.Series:
    """A column as a small set of levels: its values when few, quantile bins when numeric and many."""
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > MAX_LEVELS:
        return pd.qcut(s, q=MAX_LEVELS, duplicates="drop").astype(str)
    return s.astype(str)


@dataclass
class Tools:
    """The tools over one table. `outcome` and `treatment` are the table's own column names; `treated` is the mask of the treated
    rows when the arms are settled; `timing_of` maps a name to before, at, after or unknown; `frozen` lifts the outcome rule after
    the design is frozen; `allow_outcome_rows` are the rows on which the outcome may be joined with the arms before the freeze,
    with the words the result carries; `aliases` map the names the model knows (the pack's) to the table's columns."""

    df: pd.DataFrame
    outcome: str | None = None
    treatment: str | None = None
    treated: pd.Series | None = None
    timing_of: Mapping[str, str] | None = None
    frozen: bool = False
    allow_outcome_rows: pd.Series | None = None
    allow_outcome_words: str = "the rows where the outcome may be seen"
    aliases: Mapping[str, str] | None = None

    # ------------------------------------------------------------------ what is offered, resolving, and the rule
    def names(self) -> tuple[str, ...]:
        """The tools this table can run: the shape-aware ones only when their columns are present."""
        return tuple(n for n in NAMES if all(c in self.df.columns for c in NEEDS.get(n, ())))

    def column(self, name: str) -> str:
        """The table's column a name, a key or an alias points at."""
        for k, v in (self.aliases or {}).items():
            if (k == name or _key(str(k)) == _key(name)) and v in self.df.columns:
                return v
        if name in self.df.columns:
            return name
        for c in self.df.columns:
            if _key(str(c)) == _key(name):
                return str(c)
        raise Refused(f"refused: {name!r} is not a column of the table")

    def _need(self, name: str) -> None:
        if name not in self.names():
            raise Refused(f"refused: {name} needs columns this table does not have ({', '.join(NEEDS.get(name, ()))})")

    def _scope(self, cols: list[str], *, by_arm: bool) -> tuple[pd.DataFrame, pd.Series | None, str]:
        """The rows a tool may read: every row, or, when it would join the outcome with the arms before the freeze, the allowed rows
        only, with the words to say so; refused when there are none."""
        joins = not self.frozen and self.outcome is not None and self.outcome in cols and (by_arm or self.treatment in cols)
        if not joins:
            return self.df, self.treated, ""
        if self.allow_outcome_rows is None:
            raise Refused(
                f"refused: that would join the outcome {self.outcome!r} with the treatment before the design is frozen; "
                "the effect is the run's to find, not a fact to choose the design by"
            )
        m = self.allow_outcome_rows.reindex(self.df.index).fillna(False).astype(bool)
        if not m.any():
            raise Refused(f"refused: the outcome {self.outcome!r} may be joined with the treatment only on {self.allow_outcome_words}, and there are none")
        return self.df[m], (self.treated[m] if self.treated is not None else None), f"; {self.allow_outcome_words} only"

    @staticmethod
    def _arms(treated: pd.Series | None) -> pd.Series:
        if treated is None:
            raise Refused("refused: which rows are treated is not settled, so there are no arms")
        return treated

    # ------------------------------------------------------------------ the tools on any table
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
        df, treated, note = self._scope([c], by_arm=True)
        treated = self._arms(treated)
        text, v = FX.by_arm(df[c], treated, c)
        return Result(text + note, v)

    def association(self, a: str, b: str) -> Result:
        """How two columns move together: Pearson's r for two numbers, Cramér's V for two categories, the correlation ratio for one of each."""
        x, y = self.column(a), self.column(b)
        if x == y:
            raise Refused("refused: the same column twice")
        df, _, note = self._scope([x, y], by_arm=False)
        v, how = FX.association(df[x], df[y])
        if v is None:
            return Result(f"{x!r} and {y!r}: {how} not computable (a constant column or too few rows){note}")
        return Result(f"{x!r} and {y!r}: {how} {v:.2f}{note}", v)

    def redundancy(self, a: str, b: str) -> Result:
        """Whether two columns carry the same information or one sits inside the other: every value of one maps to one value of the other."""
        x, y = self.column(a), self.column(b)
        if x == y:
            raise Refused("refused: the same column twice")
        df, _, note = self._scope([x, y], by_arm=False)
        r = FX.redundancy(df[x], df[y])
        if r == "same":
            return Result(f"{x!r} and {y!r} carry the same information: each value of one maps to one value of the other{note}", 1.0)
        if r == "a in b":
            return Result(f"{x!r} sits inside {y!r}: every value of {x!r} maps to one value of {y!r}, and {y!r} has fewer levels{note}", 1.0)
        if r == "b in a":
            return Result(f"{y!r} sits inside {x!r}: every value of {y!r} maps to one value of {x!r}, and {x!r} has fewer levels{note}", 1.0)
        return Result(f"{x!r} and {y!r}: neither determines the other{note}", 0.0)

    def cells(self, columns: list[str]) -> Result:
        """Rows per arm in every cell of these columns, the overlap table; a number with many values is cut into six bins. At most three columns."""
        cols = [self.column(c) for c in columns]
        if not cols:
            raise Refused("refused: name at least one column")
        if len(cols) > MAX_CELL_COLUMNS:
            raise Refused(f"refused: at most {MAX_CELL_COLUMNS} columns in one table")
        df, treated, note = self._scope(cols, by_arm=True)
        treated = self._arms(treated)
        lv = pd.concat([_levels(df[c]).rename(c) for c in cols], axis=1)
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
        return Result(head + ". " + "; ".join(lines) + more + note, float(smallest))

    def timing(self) -> Result:
        """Every column's place in time against the treatment, as the pack settled it, and which are unknown."""
        when: dict[str, list[str]] = {"before": [], "at": [], "after": [], "unknown": []}
        for c in self.df.columns:
            if c in (self.outcome, self.treatment):
                continue
            w = (self.timing_of or {}).get(str(c), "unknown")
            when[w if w in when else "unknown"].append(str(c))
        return Result("; ".join(f"{k}: {', '.join(v) if v else 'none'}" for k, v in when.items()), float(len(when["unknown"])))

    # ------------------------------------------------------------------ the tools on a panel
    def by_group_over_time(self, column: str) -> Result:
        """One column by group and period on a panel: the mean per period among the units that got the change and among the others, and how the gap between them moved."""
        self._need("by_group_over_time")
        c = self.column(column)
        df, _, note = self._scope([c], by_arm=True)
        arm = df["treated"].astype(int) == 1
        s = df[c]
        numeric = pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 2
        if numeric:
            val = pd.to_numeric(s, errors="coerce")
            what = "mean"
        else:
            top = str(s.astype(str).value_counts().index[0])
            val = (s.astype(str) == top).astype(float)
            what = f"share of {top!r}"
        g = val.groupby([df["time"], arm]).mean().unstack(level=1)
        g = g.reindex(columns=[True, False])
        periods = list(g.index)
        shown = periods[:MAX_PERIODS]
        parts = []
        for p in shown:
            a, b = g.loc[p, True], g.loc[p, False]
            parts.append(f"{_period(p)}: treated {a:.3g}, others {b:.3g}" if pd.notna(a) and pd.notna(b) else f"{_period(p)}: one group absent")
        gaps = (g[True] - g[False]).dropna()
        move = float(gaps.iloc[-1] - gaps.iloc[0]) if len(gaps) >= 2 else 0.0
        tail = f"; the gap moved from {gaps.iloc[0]:+.3g} to {gaps.iloc[-1]:+.3g}" if len(gaps) >= 2 else ""
        more = f"; {len(periods) - MAX_PERIODS} more periods" if len(periods) > MAX_PERIODS else ""
        return Result(f"{c!r} by group and period ({what}): " + "; ".join(parts) + more + tail + note, move)

    def composition(self) -> Result:
        """Who is in the panel when: units present per group in each period, and how many entered after the first period or left before the last."""
        self._need("composition")
        df = self.df
        arm = df["treated"].astype(int) == 1
        per = df.groupby([df["time"], arm])["unit"].nunique().unstack(level=1).reindex(columns=[True, False]).fillna(0).astype(int)
        periods = list(per.index)
        first, last = periods[0], periods[-1]
        at = {p: set(df.loc[df["time"] == p, "unit"].astype(str)) for p in (first, last)}
        every = set(df["unit"].astype(str))
        entries = len(every - at[first])
        exits = len(every - at[last])
        shown = per.head(MAX_PERIODS)
        more = f"; {len(periods) - MAX_PERIODS} more periods" if len(periods) > MAX_PERIODS else ""
        text = (
            f"units present per period: treated {', '.join(str(int(v)) for v in shown[True])}{more}; "
            f"others {', '.join(str(int(v)) for v in shown[False])}{more}; "
            f"{entries} entered after the first period, {exits} left before the last"
        )
        return Result(text, float(entries + exits))

    # ------------------------------------------------------------------ the tools on a recentred score
    def by_side_near(self, column: str, width: float) -> Result:
        """One column on each side of the line among the rows within `width` of it, in the score's units: means and the standardised difference for a number, shares for a category."""
        self._need("by_side_near")
        c = self.column(column)
        try:
            w = float(width)
        except (TypeError, ValueError) as e:
            raise Refused("refused: width must be a number in the score's units") from e
        if not w > 0:
            raise Refused("refused: width must be a positive number in the score's units")
        df, _, note = self._scope([c], by_arm=True)
        near = df[df["x"].abs() < w]
        side = near["x"] >= 0
        if near.empty or side.all() or not side.any():
            raise Refused(f"refused: within {w:g} of the line there are no rows on one side; widen the band")
        text, v = FX.by_arm(near[c], side, c)
        head = f"within {w:g} of the line ({int((~side).sum())} rows on the control side, {int(side.sum())} on the treated side): "
        return Result(head + text.replace("among the treated", "on the treated side").replace("among the others", "on the control side") + note, v)

    def score_histogram(self, bins: int = 10) -> Result:
        """How the score is spread either side of the line: rows per bin of equal width from the farthest control-side value to the farthest treated-side value, the line as an edge."""
        self._need("score_histogram")
        try:
            k = int(bins)
        except (TypeError, ValueError) as e:
            raise Refused("refused: bins must be a whole number") from e
        k = max(2, min(MAX_BINS, k))
        k += k % 2
        x = pd.to_numeric(self.df["x"], errors="coerce").dropna()
        if x.empty:
            raise Refused("refused: the score has no values")
        reach = float(max(abs(x.min()), abs(x.max()))) or 1.0
        edges = np.linspace(-reach, reach, k + 1)
        counts, _ = np.histogram(x, bins=edges)
        half = k // 2
        left, right = counts[:half], counts[half:]
        ratio = float(right[0] / left[-1]) if left[-1] > 0 else float("inf") if right[0] > 0 else 1.0
        text = (
            f"rows per bin of width {2 * reach / k:.3g}, control side then treated side, the line between them: "
            + ", ".join(str(int(v)) for v in left)
            + " | "
            + ", ".join(str(int(v)) for v in right)
            + f"; the bin just above the line holds {ratio:.2f} times the rows of the bin just below"
        )
        return Result(text, ratio)

    # ------------------------------------------------------------------ for the episode
    def call(self, name: str, args: dict[str, Any]) -> Result:
        """Run one tool by name; a wrong name or wrong arguments are a refusal like any other."""
        if name not in self.names():
            raise Refused(f"refused: no tool named {name!r}; the tools are {', '.join(self.names())}")
        try:
            return getattr(self, name)(**args)
        except TypeError as e:
            raise Refused(f"refused: {name} could not run with {args!r}: {e}") from e

    def schemas(self) -> list[StructuredTool]:
        """The tools as the model sees them, bound to this table."""
        return [StructuredTool.from_function(func=getattr(self, n), name=n, description=str(getattr(Tools, n).__doc__)) for n in self.names()]
