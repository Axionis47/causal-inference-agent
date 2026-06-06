"""Tool registration index for the effect estimator agent.

Order matches the prompt's workflow: data summary, balance check,
propensity scores, run one or more methods, diagnostics, cross-method
comparison, finalize.
"""

from . import (
    check_covariate_balance,
    check_method_diagnostics,
    compare_estimates,
    estimate_propensity_scores,
    finalize_estimation,
    get_data_summary,
    run_estimation_method,
)

MODULES = [
    get_data_summary,
    check_covariate_balance,
    estimate_propensity_scores,
    run_estimation_method,
    check_method_diagnostics,
    compare_estimates,
    finalize_estimation,
]

__all__ = ["MODULES"]
