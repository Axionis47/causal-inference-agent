"""Tool registration index for the sensitivity analyst agent.

Dispatch order reflects the prompt's workflow: estimates summary first,
then sensitivity tests in the order the prompt lists them, then
finalize.
"""

from . import (
    check_variance_stability,
    compute_e_value,
    compute_rosenbaum_bounds,
    finalize_sensitivity,
    get_estimates_summary,
    run_placebo_test,
    run_specification_curve,
    run_subgroup_analysis,
)

MODULES = [
    get_estimates_summary,
    compute_e_value,
    compute_rosenbaum_bounds,
    run_specification_curve,
    run_placebo_test,
    run_subgroup_analysis,
    check_variance_stability,
    finalize_sensitivity,
]

__all__ = ["MODULES"]
