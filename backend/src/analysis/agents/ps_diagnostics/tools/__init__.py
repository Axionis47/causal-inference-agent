"""Tool registration index for the propensity score diagnostics agent.

Order is the dispatch order the prompt walks: estimate PS, then run the
three diagnostic checks (overlap, balance, calibration), then the
optional trimming / per-covariate inspection tools, then finalize.
"""

from . import (
    check_balance,
    check_calibration,
    check_overlap,
    compute_trimmed,
    estimate_ps,
    finalize_diagnostics,
    get_covariate_summary,
)

MODULES = [
    estimate_ps,
    check_overlap,
    check_balance,
    check_calibration,
    compute_trimmed,
    get_covariate_summary,
    finalize_diagnostics,
]

__all__ = ["MODULES"]
