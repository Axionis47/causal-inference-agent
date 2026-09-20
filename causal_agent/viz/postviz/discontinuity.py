"""The discontinuity lane's own figures, pure over its artifacts and the tables it wrote: the jump at the cutoff (the binned
means with a local fit on each side, recomputed by code from the design's own bandwidth and order), the score's density
either side, each covariate's jump at the cutoff, the estimate across bandwidths, and the placebo cutoffs. Every drawn value
rests on an address the run produced."""

from __future__ import annotations

import numpy as np
import pandas as pd

from causal_agent.viz.spec import FigureSpec, Mark, Series

GRID = 40


def _fit_side(x: np.ndarray, y: np.ndarray, h: float, p: int, treated: bool) -> tuple[list[float], list[float]] | None:
    """A triangular-kernel weighted polynomial of order p on one side within h, evaluated on a grid up to the cutoff."""
    mask = ((x >= 0) & (x < h)) if treated else ((x < 0) & (x > -h))
    xs, ys = x[mask], y[mask]
    if len(xs) <= p + 1:
        return None
    w = np.sqrt(np.clip(1 - np.abs(xs) / h, 0, None))
    try:
        coef = np.polyfit(xs, ys, p, w=w)
    except Exception:
        return None
    grid = np.linspace(0, h, GRID) if treated else np.linspace(-h, 0, GRID)
    return [float(g) for g in grid], [float(v) for v in np.polyval(coef, grid)]


def rd_plot(bins: pd.DataFrame | None, canon: pd.DataFrame, h: float, p: int, contrast: str, score_name: str = "score") -> FigureSpec | None:
    """The binned means of the outcome against the score (from the library's bins) with a local fit on each side of the cutoff."""
    if canon is None or not {"x", "y"} <= set(canon.columns) or canon.empty or not h or h <= 0:
        return None
    x, y = canon["x"].to_numpy(dtype=float), canon["y"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    series: list[Series] = []
    if bins is not None and {"rdplot_mean_x", "rdplot_mean_y"} <= set(bins.columns):
        b = bins.dropna(subset=["rdplot_mean_x", "rdplot_mean_y"])
        series.append(Series(name="binned means", x=[float(v) for v in b["rdplot_mean_x"]], y=[float(v) for v in b["rdplot_mean_y"]],
                             n=[int(v) for v in b["rdplot_N"]] if "rdplot_N" in b.columns else None))
    left, right = _fit_side(x, y, h, p, False), _fit_side(x, y, h, p, True)
    if left:
        series.append(Series(name="fit, control side", x=left[0], y=left[1]))
    if right:
        series.append(Series(name="fit, treated side", x=right[0], y=right[1]))
    if not series:
        return None
    jump = (right[1][0] - left[1][-1]) if left and right else None
    note = f"the two fits within h = {h:.3g} meet the cutoff {jump:+.3g} apart" if jump is not None else "one side has too few rows within the bandwidth for a fit"
    return FigureSpec(id=f"rd_plot_{contrast}", kind="points", title="The outcome against the score, either side of the cutoff", x_label=f"{score_name}, distance from the cutoff", y_label="outcome",
                      series=series, marks=[Mark(kind="vline", at=0.0, label="the cutoff")], note=note, draws_on=["design.bandwidth", f"estimate:{contrast}.value"])


def density_test(x_all: pd.Series | np.ndarray, facts: dict | None, contrast: str, bins: int = 30, sampled_by_side: bool = False) -> FigureSpec | None:
    """How many units sit at each score either side of the cutoff, with the manipulation test's verdict in the note."""
    x = np.asarray(x_all, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return None
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return None
    width = (hi - lo) / bins
    edges = np.concatenate([np.arange(0, lo - width, -width)[::-1], np.arange(width, hi + width, width)])  # an edge at the cutoff itself
    counts, edges = np.histogram(x, bins=edges)
    centres = [(float(edges[i]) + float(edges[i + 1])) / 2 for i in range(len(counts))]
    left = Series(name="below the cutoff", x=[c for c in centres if c < 0], y=[float(n) for c, n in zip(centres, counts) if c < 0])
    right = Series(name="at or above the cutoff", x=[c for c in centres if c >= 0], y=[float(n) for c, n in zip(centres, counts) if c >= 0])
    f = facts or {}
    if sampled_by_side:
        note = "the rows were drawn by side of the cutoff, so their density says nothing about manipulation"
    elif f.get("computable"):
        note = f"density {f['hat_left']:.3g} just below and {f['hat_right']:.3g} just above; test p = {f['p']:.3g}" + (" (a jump)" if f["p"] < 0.10 else " (no sign of bunching)")
    else:
        note = "the manipulation test could not be read here: " + str(f.get("reason") or "not run")
    return FigureSpec(id=f"density_{contrast}", kind="density", title="How many units sit at each score", x_label="distance from the cutoff", y_label="units",
                      series=[left, right], marks=[Mark(kind="vline", at=0.0, label="the cutoff")], note=note, draws_on=[f"check:{contrast}.density"])


def covariate_continuity(per: dict[str, dict] | None, contrast: str, names: dict[str, str] | None = None, threshold: float = 0.05) -> FigureSpec | None:
    """Each predetermined covariate's jump at the cutoff with its interval; a jump the design says should not be there."""
    if not per:
        return None
    names = names or {}
    cols = list(per)
    failed = [names.get(c, c) for c in cols if per[c].get("p") is not None and per[c]["p"] < threshold]
    return FigureSpec(id=f"continuity_{contrast}", kind="interval", title="Each covariate's jump at the cutoff", x_label="", y_label="jump at the cutoff",
                      series=[Series(name="jump", x=[names.get(c, c) for c in cols], y=[per[c].get("jump") for c in cols], lo=[per[c].get("ci_low") for c in cols], hi=[per[c].get("ci_high") for c in cols])],
                      marks=[Mark(kind="hline", at=0.0, label="no jump")],
                      note=(", ".join(failed) + " differ at the cutoff") if failed else "every covariate is continuous at the cutoff",
                      draws_on=[f"check:{contrast}.covariate_continuity"])


def bandwidth_curve(points: list[dict] | None, primary_h: float | None, contrast: str) -> FigureSpec | None:
    """The estimate at every bandwidth the falsification tried, with the one the design used marked."""
    pts = sorted((p for p in points or [] if "value" in p and p.get("at") is not None), key=lambda p: p["at"])
    if not pts:
        return None
    marks = [Mark(kind="vline", at=float(primary_h), label="h used")] if primary_h else []
    return FigureSpec(id=f"bandwidths_{contrast}", kind="interval", title="The estimate across bandwidths", x_label="bandwidth", y_label="effect",
                      series=[Series(name="estimate", x=[float(p["at"]) for p in pts], y=[p["value"] for p in pts], lo=[p["lo"] for p in pts], hi=[p["hi"] for p in pts], n=[int(p["n_l"] + p["n_r"]) for p in pts])],
                      marks=marks + [Mark(kind="hline", at=0.0, label="no effect")],
                      note="the conclusion should not depend on the bandwidth" + ("; a point marked uninformative had too few rows" if any(not p.get("informative") for p in pts) else ""),
                      draws_on=[f"placebo:{contrast}.bandwidth_grid.detail", "design.bandwidth"])


def placebo_cutoffs(points: list[dict] | None, primary: dict | None, contrast: str) -> FigureSpec | None:
    """The jump at each artificial cutoff beside the jump at the real one; only the real one should show a jump."""
    pts = [p for p in points or [] if "value" in p]
    if not pts or not primary:
        return None
    rows = sorted([(p.get("at", 0.0), p["label"], p["value"], p["lo"], p["hi"]) for p in pts] + [(0.0, "the cutoff", primary.get("value"), primary.get("ci_low"), primary.get("ci_high"))], key=lambda r: r[0])
    return FigureSpec(id=f"placebo_cutoffs_{contrast}", kind="interval", title="The jump at the real cutoff and at artificial ones", x_label="", y_label="jump",
                      series=[Series(name="jump", x=[r[1] for r in rows], y=[r[2] for r in rows], lo=[r[3] for r in rows], hi=[r[4] for r in rows])],
                      marks=[Mark(kind="hline", at=0.0, label="no jump")], note="an artificial cutoff should show no jump",
                      draws_on=[f"placebo:{contrast}.placebo_cutoffs.detail", f"estimate:{contrast}.value"])
