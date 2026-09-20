"""The diff-in-diff lane's own figures, pure over its panel and artifacts: the two groups' paths with the treated group's
path had the change not happened, and the spread of the placebo effects against the observed one. The event study is in
`common`. Every drawn value rests on an address the run produced."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_agent.viz.spec import FigureSpec, Mark, Series


def _x(v) -> str | float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def paths_with_counterfactual(panel: pd.DataFrame, contrast: str, estimate: float | None = None) -> FigureSpec | None:
    """The mean outcome by period for the group that got the change and the group that did not, and the treated group's path
    without the change: its own pre-period level plus the comparison group's movement. The gap after the change is the design."""
    need = {"y", "time", "treated", "post"}
    if panel is None or not need <= set(panel.columns) or panel.empty:
        return None
    by = panel.groupby(["time", "treated"])["y"].agg(["mean", "size"]).reset_index()
    times = sorted(panel["time"].unique())
    pre = panel[panel["post"] == 0]
    if pre.empty:
        return None
    t_pre = float(pre.loc[pre["treated"] == 1, "y"].mean())
    c_pre = float(pre.loc[pre["treated"] == 0, "y"].mean())
    first_post = min(panel.loc[panel["post"] == 1, "time"].unique(), default=None)

    def series(name: str, flag: int) -> Series:
        rows = by[by["treated"] == flag].set_index("time")
        return Series(name=name, x=[_x(t) for t in times], y=[float(rows.loc[t, "mean"]) if t in rows.index else None for t in times],
                      n=[int(rows.loc[t, "size"]) if t in rows.index else 0 for t in times])

    got, not_ = series("got the change", 1), series("did not", 0)
    cf = Series(name="the treated group without the change", x=[_x(t) for t in times],
                y=[(t_pre + (c - c_pre)) if (p and c is not None) else None for t, c, p in zip(times, not_.y, [bool(panel.loc[panel["time"] == t, "post"].max()) for t in times])])
    marks = [Mark(kind="vline", at=_x(first_post), label="the change")] if first_post is not None else []
    gap = None
    if estimate is not None:
        gap = f"the design's estimate is {estimate:.3g}, the gap between the treated path and its path without the change"
    return FigureSpec(id=f"paths_{contrast}", kind="lines", title="The two groups over time, and the treated group's path without the change", x_label="period", y_label="outcome",
                      series=[got, not_, cf], marks=marks, note=gap or "the two paths before the change are the parallel-paths assumption made visible",
                      draws_on=["design.periods"] + ([f"estimate:{contrast}.value"] if estimate is not None else []))


def placebo_distribution(draws: list[float], observed: float | None, p: float | None, contrast: str, bins: int = 30) -> FigureSpec | None:
    """Where the observed effect sits among the effects from reassigning the treated label at random."""
    vals = [float(v) for v in draws or [] if v is not None and np.isfinite(v)]
    if len(vals) < 5:
        return None
    lo, hi = min(vals + ([observed] if observed is not None else [])), max(vals + ([observed] if observed is not None else []))
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    centres = [(float(edges[i]) + float(edges[i + 1])) / 2 for i in range(len(counts))]
    marks = [Mark(kind="vline", at=float(observed), label="observed")] if observed is not None else []
    share = f"share of reassignments with an effect at least as large: {p:.2f}" if p is not None else "no p-value"
    return FigureSpec(id=f"placebo_{contrast}", kind="density", title="The observed effect among effects from reassigning the treated label", x_label="placebo effect", y_label="count",
                      series=[Series(name="placebo effects", x=centres, y=[float(c) for c in counts])], marks=marks, note=f"{len(vals)} reassignments; {share}",
                      draws_on=[f"placebo:{contrast}.placebo_group.p_value"] + ([f"estimate:{contrast}.value"] if observed is not None else []))
