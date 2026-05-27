"""Tool registration index for the data_repair agent.

Each module exposes a SCHEMA dict and an async handle(agent, state, **kwargs)
function. The agent iterates MODULES at init time.
"""

from . import (
    check_collinearity,
    check_missing_values,
    check_outliers,
    check_skewness,
    finalize_repairs,
    get_data_summary,
    repair_collinearity,
    repair_missing,
    repair_outliers,
    validate_current_state,
)

MODULES = [
    get_data_summary,
    check_missing_values,
    check_outliers,
    check_collinearity,
    check_skewness,
    repair_missing,
    repair_outliers,
    repair_collinearity,
    validate_current_state,
    finalize_repairs,
]

__all__ = ["MODULES"]
