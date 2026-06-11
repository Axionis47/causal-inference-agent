"""Deterministic dataset profiling for the data-review gate.

Pure pandas/numpy, no LLM. Re-homed from the deleted analysis package; this is
the facts-only profile shown before the gate.
"""
from src.profiling.profile import (
    compute_basic_profile,
    compute_deterministic_profile,
    detect_time_column,
)

__all__ = [
    "compute_basic_profile",
    "compute_deterministic_profile",
    "detect_time_column",
]
