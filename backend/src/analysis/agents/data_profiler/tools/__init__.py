"""Tool registry for the data_profiler agent."""

from . import (
    analyze_column,
    check_discontinuity,
    check_relationship,
    check_time_dimension,
    check_treatment_balance,
    finalize_profile,
    get_overview,
)

MODULES = [
    get_overview,
    analyze_column,
    check_treatment_balance,
    check_relationship,
    check_time_dimension,
    check_discontinuity,
    finalize_profile,
]
