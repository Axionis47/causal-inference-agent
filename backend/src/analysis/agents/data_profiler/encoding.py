"""Treatment encoding helpers for the data_profiler agent.

Deterministic logic for choosing an encoding strategy, picking a control
value, and constructing a TreatmentEncoding.
"""
from __future__ import annotations

import pandas as pd

from .output import TreatmentEncoding

CONTROL_KEYWORDS = ("control", "no", "placebo", "none", "baseline", "reference")


def detect_strategy(values: list) -> str | None:
    """Choose encoding strategy based on number of unique values."""
    n = len(values)
    if n == 2:
        return "label_encode"
    if 2 < n <= 10:
        return "collapse_to_binary"
    return None


def detect_control_value(values: list, value_counts: pd.Series) -> str:
    """Pick the value most likely to be the control group.

    Searches for known control keywords first; falls back to the most
    frequent value when no keyword matches.
    """
    for v in values:
        if any(kw in str(v).lower() for kw in CONTROL_KEYWORDS):
            return str(v)
    return str(value_counts.idxmax())


def build_encoding(
    values: list, strategy: str, control: str | None = None
) -> TreatmentEncoding | None:
    """Build a TreatmentEncoding from observed values and a strategy.

    Returns None for unsupported strategies.
    """
    if strategy not in ("label_encode", "collapse_to_binary"):
        return None

    value_mapping = None
    if values:
        if strategy == "collapse_to_binary" and control:
            value_mapping = {v: 0 if str(v) == str(control) else 1 for v in values}
        elif strategy == "label_encode" and len(values) == 2:
            value_mapping = {values[0]: 0, values[1]: 1}

    return TreatmentEncoding(
        original_type="multi_categorical" if strategy == "collapse_to_binary" else "binary",
        strategy=strategy,
        control_value=control,
        value_mapping=value_mapping,
    )


def auto_encode(values: list, value_counts: pd.Series) -> TreatmentEncoding | None:
    """Detect strategy and build encoding automatically from data."""
    strategy = detect_strategy(values)
    if not strategy:
        return None
    control = (
        str(values[0])
        if strategy == "label_encode"
        else detect_control_value(values, value_counts)
    )
    return build_encoding(values, strategy, control)
