"""Tool registration index for the critique agent.

Critique uses BaseAgent + a custom agentic loop (not ReActAgent). The
loop reads `SCHEMAS` as the LLM function-calling tool list, and
dispatches investigation tools through `BY_NAME`. The `finalize_critique`
tool has a schema entry but no handler: the loop short-circuits on
that tool name and returns the LLM's args as the final result.
"""

from . import (
    check_covariate_balance,
    check_estimate_consistency,
    check_influential_observations,
    check_sensitivity_robustness,
    check_subgroup_effects,
    finalize_critique,
    get_analysis_summary,
    verify_propensity_scores,
)

SCHEMAS = [
    get_analysis_summary.SCHEMA,
    check_covariate_balance.SCHEMA,
    check_estimate_consistency.SCHEMA,
    check_subgroup_effects.SCHEMA,
    check_influential_observations.SCHEMA,
    verify_propensity_scores.SCHEMA,
    check_sensitivity_robustness.SCHEMA,
    finalize_critique.SCHEMA,
]

BY_NAME = {
    "get_analysis_summary": get_analysis_summary,
    "check_covariate_balance": check_covariate_balance,
    "check_estimate_consistency": check_estimate_consistency,
    "check_subgroup_effects": check_subgroup_effects,
    "check_influential_observations": check_influential_observations,
    "verify_propensity_scores": verify_propensity_scores,
    "check_sensitivity_robustness": check_sensitivity_robustness,
}

__all__ = ["SCHEMAS", "BY_NAME"]
