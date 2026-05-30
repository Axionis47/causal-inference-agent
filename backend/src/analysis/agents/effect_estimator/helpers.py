"""Pure helpers for the effect estimator: prompts, OLS fallback, key-diagnostic extraction.

Helpers here are stateless or take an agent reference. The DAG-aware
pieces (compute_adjustment_set / classify_causal_role) live in
`dag_utils.py`; pair selection lives in `pair_selection.py`;
covariate selection in `covariates.py`. This file holds the bits
that didn't fit those buckets.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.analysis.agents.base import (
    AnalysisState,
    ToolResultStatus,
    TreatmentEffectResult,
)

from .estimation_methods import run_method


def load_dataframe(dataframe_path: str | None) -> pd.DataFrame | None:
    """Read the parquet at dataframe_path; raise on read failure (caller decides)."""
    if dataframe_path is None:
        return None
    return pd.read_parquet(dataframe_path)


def build_initial_observation(
    treatment_var: str | None,
    outcome_var: str | None,
    covariates: list[str],
    n_samples: int | str,
) -> str:
    """Lean initial observation for the ReAct loop."""
    cov_tail = "..." if len(covariates) > 20 else ""
    return f"""You are estimating treatment effects for a causal analysis.

Treatment variable: {treatment_var}
Outcome variable: {outcome_var}
Available covariates: {covariates[:20]}{cov_tail}
Number of samples: {n_samples}

BEFORE RUNNING METHODS, pull context you need:
- Call get_dag_adjustment_set to get the proper covariate adjustment set
- Call get_confounder_analysis for ranked confounders
- Call ask_domain_knowledge with "Are there any immutable or post-treatment variables?"

Then follow the estimation workflow:
Step 1: Call run_estimation_method with method="ols"
Step 2: Call run_estimation_method with method="ipw"
Step 3: Call run_estimation_method with method="aipw"
Step 4: Call compare_estimates to see results
Step 5: Call finalize_estimation with your best estimate

You MUST attempt ALL three methods (OLS, IPW, AIPW). If a method fails due to
missing covariates or other issues, the tool will return an error - that is fine,
just proceed to the next method.

START NOW: First call get_dag_adjustment_set, then proceed to estimation."""


def build_react_prompt(
    steps,
    step_num: int,
    max_steps: int,
    observation: str,
    n_results: int,
) -> str:
    """Compose the per-step ReAct prompt with urgency cues.

    Reads only the immutable view of prior steps (last 3) and the
    counts; no agent-internal state mutation.
    """
    prompt = f"""Step {step_num}/{max_steps}

OBSERVATION:
{observation}

"""
    if steps:
        prompt += "PREVIOUS STEPS:\n"
        for step in steps[-3:]:
            ok = step.result and step.result.status == ToolResultStatus.SUCCESS
            prompt += f"Step {step.step_number}: Action={step.action}, "
            prompt += f"Result={'Success' if ok else 'Failed'}\n"
        prompt += "\n"

    has_run_estimation = any(s.action == "run_estimation_method" for s in steps)

    if not has_run_estimation and step_num >= 3:
        prompt += """⚠️ URGENT: You have NOT called run_estimation_method yet!
You MUST call run_estimation_method with method="ols" RIGHT NOW.
Do NOT call any more information-gathering tools.

"""
    elif not has_run_estimation and step_num >= 2:
        prompt += """REMINDER: Call run_estimation_method with method="ols" on this step.
Information gathering is complete enough to proceed.

"""
    elif has_run_estimation and n_results >= 2:
        prompt += """You have multiple estimates. Consider calling finalize_estimation or compare_estimates.

"""

    prompt += """Based on the observation, think step by step:
1. What does this observation tell me?
2. What should I do next to make progress?
3. Which tool should I use?

Then call the appropriate tool."""

    return prompt


def auto_finalize(
    df: pd.DataFrame | None,
    state: AnalysisState | None,
    treatment_var: str | None,
    outcome_var: str | None,
    covariates: list[str],
    results: list[TreatmentEffectResult],
    logger,
) -> dict[str, Any]:
    """Synthesize a finalize result when the LLM exits without calling finalize_estimation.

    Emergency fallback: if no methods ran, fit OLS so the pipeline
    always has something to report. Then pick AIPW / doubly-robust
    if available, else the median estimate.
    """
    if not results and df is not None and treatment_var and outcome_var:
        try:
            ols_result = run_method(
                "ols", treatment_var, outcome_var,
                covariates or [], df, state,
            )
            if ols_result:
                results.append(ols_result)
                logger.info(
                    "auto_finalize_ols_fallback",
                    estimate=ols_result.estimate,
                )
        except Exception as e:
            logger.warning("auto_finalize_ols_failed", error=str(e))

    if not results:
        return {
            "preferred_method": "none",
            "preferred_estimate": 0.0,
            "confidence_level": "low",
            "reasoning": "No estimation methods succeeded",
            "results": [],
        }

    preferred = None
    for r in results:
        if "doubly" in r.method.lower() or "aipw" in r.method.lower():
            preferred = r
            break

    if not preferred:
        estimates = sorted(results, key=lambda x: x.estimate)
        preferred = estimates[len(estimates) // 2]

    return {
        "preferred_method": preferred.method,
        "preferred_estimate": preferred.estimate,
        "confidence_level": "medium" if len(results) >= 2 else "low",
        "reasoning": f"Auto-finalized with {len(results)} methods. Preferred: {preferred.method}",
        "results": results,
    }


_METHOD_DIAGNOSTIC_KEYS: dict[str, list[str]] = {
    "ols": [
        "r_squared", "adj_r_squared", "reset_f_pval",
        "vif_warning", "vif_max", "n_covariates",
    ],
    "ipw": [
        "ess_treated", "ess_control", "ps_mean",
        "n_extreme_weights_treated", "n_extreme_weights_control",
        "weight_trimming", "overlap_quality",
    ],
    "aipw": [
        "ps_overlap", "outcome_r2_treated", "outcome_r2_control",
        "cross_fitting_folds", "overlap_quality",
    ],
    "psm": [
        "n_matched", "n_unmatched", "match_rate",
        "common_support", "ps_mean_treated", "ps_mean_control",
    ],
    "dml": [
        "n_folds", "ml_method", "outcome_residual_var",
        "treatment_residual_var", "residual_correlation",
    ],
    "iv": [
        "first_stage_f_partial", "weak_instrument_severe",
        "hansen_j_pvalue", "endogeneity_dwh_pvalue",
        "treatment_endogenous", "instruments",
    ],
    "rdd": [
        "bandwidth", "effective_n_left", "effective_n_right",
        "mccrary_pvalue", "kernel", "cutoff",
    ],
    "did": [
        "parallel_trends_pvalue", "n_pre_periods", "n_post_periods",
        "used_median_cutoff",
    ],
}


def extract_key_diagnostics(method_name: str, diagnostics: dict | None) -> dict:
    """Compact per-method-family diagnostics for trace visibility."""
    if not diagnostics:
        return {}

    key: dict = {}
    method_lower = method_name.lower().replace("-", "_").replace(" ", "_")

    matched_keys: list[str] = []
    for family, keys in _METHOD_DIAGNOSTIC_KEYS.items():
        if family in method_lower:
            matched_keys = keys
            break

    for field_name in matched_keys:
        if field_name in diagnostics:
            val = diagnostics[field_name]
            if isinstance(val, float):
                key[field_name] = round(val, 4)
            else:
                key[field_name] = val

    return key
