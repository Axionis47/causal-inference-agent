"""Profiling plots rendered to PNG bytes. Headless, deterministic."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis_v2.spec import ProfileSummary  # noqa: E402

MAX_BARS = 20
MAX_HISTS = 12


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def missingness_png(summary: ProfileSummary) -> bytes | None:
    """Horizontal bars of missing fraction for columns that have any."""
    cols = [c for c in summary.columns if c.missing_count > 0]
    if not cols:
        return None
    cols = sorted(cols, key=lambda c: c.missing_fraction, reverse=True)[:MAX_BARS]
    fig, ax = plt.subplots(figsize=(7, max(2.0, 0.35 * len(cols))))
    ax.barh([c.name for c in cols], [c.missing_fraction for c in cols], color="#b45f5f")
    ax.set_xlabel("missing fraction")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("Missing data by column")
    return _to_png(fig)


def numeric_distributions_png(df: pd.DataFrame, summary: ProfileSummary) -> bytes | None:
    """Histogram grid for numeric (non-id) columns, up to MAX_HISTS."""
    numeric = [
        c.name
        for c in summary.columns
        if c.semantic_type in ("numeric", "ordinal") and not c.is_constant
    ][:MAX_HISTS]
    if not numeric:
        return None
    n = len(numeric)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows))
    flat = axes.flatten() if n > 1 else [axes]
    for ax, name in zip(flat, numeric):
        df[name].dropna().hist(ax=ax, bins=30, color="#5f7fb4")
        ax.set_title(name, fontsize=9)
    for ax in flat[n:]:
        ax.set_visible(False)
    fig.suptitle("Numeric column distributions", fontsize=11)
    return _to_png(fig)
