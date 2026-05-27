"""Tool registration index for the confounder_discovery agent."""

from . import (
    compute_correlation,
    compute_partial_correlation,
    finalize_confounders,
    get_candidate_variables,
    test_confounder_criteria,
)

MODULES = [
    get_candidate_variables,
    compute_correlation,
    compute_partial_correlation,
    test_confounder_criteria,
    finalize_confounders,
]

__all__ = ["MODULES"]
