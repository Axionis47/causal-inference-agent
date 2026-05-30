"""Tool registration index for the ReAct effect estimator."""

from . import (
    analyze_treatment,
    check_assumptions,
    compare_results,
    inspect_data,
    run_method,
)

MODULES = [
    inspect_data,
    analyze_treatment,
    check_assumptions,
    run_method,
    compare_results,
]

__all__ = ["MODULES"]
