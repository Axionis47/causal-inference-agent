"""Discontinuity: the score's density around the cutoff, and the outcome by score bin. The rows a side come from the
same counts the probe uses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_agent.common.contracts import Probe
from causal_agent.viz.spec import Figure, FigureSpec, Mark, Series


def rows_by_side(x: pd.Series, cutoff: float) -> tuple[int, int]:
    return int((x < cutoff).sum()), int((x > cutoff).sum())


def side_probe(x: pd.Series, cutoff: float, score: str, floor: int) -> Probe:
    below, above = rows_by_side(x, cutoff)
    return Probe(
        family="discontinuity",
        name="rows_by_side",
        value=float(min(below, above)),
        passed=min(below, above) >= floor,
        detail=f"{below} rows below {cutoff:g} and {above} above on {score!r}; floor {floor} a side",
    )


def _bins(x: pd.Series, cutoff: float, bins: int, window: float | None) -> list[tuple[float, float]]:
    lo, hi = float(x.min()), float(x.max())
    if window is not None:
        lo, hi = max(lo, cutoff - window), min(hi, cutoff + window)
    half = max(bins // 2, 1)
    left = np.linspace(lo, cutoff, half + 1) if lo < cutoff else np.array([cutoff])
    right = np.linspace(cutoff, hi, half + 1) if hi > cutoff else np.array([cutoff])
    edges = np.unique(np.concatenate([left, right]))
    return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]


def density(
    df: pd.DataFrame, score: str | None, cutoff: float, bins: int = 20, window: float | None = None, floor: int = 20, addresses: list[str] | None = None
) -> Figure:
    """Rows per score bin either side of the cutoff: bunching shows as a jump at the line."""
    if score not in df.columns:
        return Figure.refused(f"{score!r} is not a column in the table", "discontinuity.density")
    assert score is not None
    x = pd.to_numeric(df[score], errors="coerce").dropna()
    if x.empty:
        return Figure.refused(f"{score!r} does not read as numbers", "discontinuity.density")
    probe = side_probe(x, cutoff, score, floor)
    edges = _bins(x, cutoff, bins, window)
    xs, ys = [], []
    for a, b in edges:
        inside = (x >= a) & (x < b) if b <= cutoff else (x > a) & (x <= b)
        xs.append(float((a + b) / 2))
        ys.append(float(inside.sum()))
    spec = FigureSpec(
        id=f"density_{score}".replace(" ", "_"),
        kind="density",
        title=f"How many rows sit at each {score}",
        x_label=score,
        y_label="rows",
        series=[Series(name="rows", x=xs, y=ys)],
        marks=[Mark(kind="vline", at=cutoff, label="cutoff")],
        note=f"{probe.detail}",
        draws_on=list(addresses or []) + [probe.address],
    )
    return Figure(made=True, spec=spec, probe=probe, function="discontinuity.density")


def outcome_by_bin(
    df: pd.DataFrame,
    score: str | None,
    cutoff: float,
    outcome: str | None,
    bins: int = 20,
    window: float | None = None,
    floor: int = 20,
    addresses: list[str] | None = None,
) -> Figure:
    """Mean outcome per score bin either side of the cutoff: the jump at the line is what the design estimates."""
    for c in (score, outcome):
        if c not in df.columns:
            return Figure.refused(f"{c!r} is not a column in the table", "discontinuity.outcome_by_bin")
    assert score is not None and outcome is not None
    x = pd.to_numeric(df[score], errors="coerce")
    y = pd.to_numeric(df[outcome], errors="coerce")
    keep = x.notna() & y.notna()
    x, y = x[keep], y[keep]
    if x.empty:
        return Figure.refused("no rows with both a score and an outcome", "discontinuity.outcome_by_bin")
    probe = side_probe(x, cutoff, score, floor)
    xs, ys, ns = [], [], []
    for a, b in _bins(x, cutoff, bins, window):
        inside = (x >= a) & (x < b) if b <= cutoff else (x > a) & (x <= b)
        n = int(inside.sum())
        xs.append(float((a + b) / 2))
        ys.append(float(y[inside].mean()) if n else None)
        ns.append(n)
    spec = FigureSpec(
        id=f"outcome_by_bin_{outcome}".replace(" ", "_"),
        kind="points",
        title=f"{outcome} by {score}, either side of {cutoff:g}",
        x_label=score,
        y_label=f"mean {outcome}",
        series=[Series(name=f"mean {outcome}", x=xs, y=ys, n=ns)],
        marks=[Mark(kind="vline", at=cutoff, label="cutoff")],
        note="a jump at the line is the effect the design reads; a smooth curve through it is none",
        draws_on=list(addresses or []) + [probe.address],
    )
    return Figure(made=True, spec=spec, probe=probe, function="discontinuity.outcome_by_bin")
