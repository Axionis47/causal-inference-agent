"""Pure helpers for the propensity score diagnostics agent.

DataFrame loading, the confounder priority chain
(dag_expert -> confounder_discovery -> data profile -> all numeric),
the initial-observation string, and the auto-finalize heuristic that
runs when the LLM exits the loop without calling finalize_diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def load_dataframe(dataframe_path: str | None, logger: Any) -> pd.DataFrame | None:
    """Load parquet or CSV from the state path; return None on failure."""
    if not dataframe_path:
        return None
    try:
        if dataframe_path.endswith(".csv"):
            return pd.read_csv(dataframe_path)
        return pd.read_parquet(dataframe_path)
    except Exception as e:
        logger.error("load_failed", error=str(e))
        return None


def resolve_confounders(
    state,
    df: pd.DataFrame,
    treatment_var: str,
    outcome_var: str,
) -> list[str]:
    """Priority chain: dag_expert > confounder_discovery > profile > all numeric.

    Always filters out treatment / outcome / unknown columns at the end.
    """
    confounders: list[str]
    if (
        state.proposed_dag
        and state.proposed_dag.adjustment_set
        and len(state.proposed_dag.adjustment_set) > 0
    ):
        confounders = list(state.proposed_dag.adjustment_set)
    elif (
        state.confounder_discovery
        and state.confounder_discovery.get("ranked_confounders")
    ):
        confounders = list(state.confounder_discovery["ranked_confounders"])
    elif state.data_profile and state.data_profile.potential_confounders:
        confounders = list(state.data_profile.potential_confounders)
    else:
        confounders = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != treatment_var and c != outcome_var
        ]

    return [
        c for c in confounders
        if c in df.columns and c != treatment_var and c != outcome_var
    ]


def initial_observation_text(
    treatment_var: str | None,
    outcome_var: str | None,
    confounders: list[str],
) -> str:
    """Front-load PS-specific context (treatment, outcome, full covariate list)."""
    tail = "..." if len(confounders) > 15 else ""
    return f"""PROPENSITY SCORE DIAGNOSTICS TASK
Treatment variable: {treatment_var}
Outcome variable: {outcome_var}
Covariates for PS model ({len(confounders)}): {confounders[:15]}{tail}

Systematically check:
1. Estimate propensity scores using estimate_propensity_scores
2. Check overlap between treatment groups
3. Check covariate balance (with and without IPW weighting)
4. Check model calibration

If diagnostics are poor, try alternative specifications (interactions, polynomials, trimming).
Call finalize_diagnostics when you have sufficient evidence."""


def auto_finalize(
    propensity_scores: np.ndarray | None,
    treatment: np.ndarray | None,
) -> dict:
    """Heuristic finalize when the LLM exits the loop without calling finalize_diagnostics.

    Mirrors the contract finalize_diagnostics produces: model_quality,
    proceed_with_analysis, recommended_method, trimming_bounds,
    warnings, reasoning.
    """
    if propensity_scores is None or treatment is None:
        return {
            "model_quality": "unreliable",
            "proceed_with_analysis": False,
            "recommended_method": "other",
            "warnings": ["Could not estimate propensity scores"],
            "reasoning": "PS estimation failed",
        }

    ps = propensity_scores
    T = treatment
    treated_idx = T == 1
    control_idx = T == 0

    overlap_min = max(ps[control_idx].min(), ps[treated_idx].min())
    overlap_max = min(ps[control_idx].max(), ps[treated_idx].max())
    pct_overlap = np.mean((ps >= overlap_min) & (ps <= overlap_max)) * 100

    from src.config.settings import get_settings
    _s = get_settings()
    positivity_violations = int(
        np.sum((ps < _s.ps_clip_lower) | (ps > _s.ps_clip_upper))
    )

    if pct_overlap > _s.ps_overlap_good_threshold and positivity_violations < 10:
        quality = "good"
        method = "aipw"
    elif pct_overlap > _s.ps_overlap_acceptable_threshold:
        quality = "acceptable"
        method = "aipw"
    else:
        quality = "needs_improvement"
        method = "doubly_robust"

    return {
        "model_quality": quality,
        "proceed_with_analysis": True,
        "recommended_method": method,
        "trimming_bounds": [0.01, 0.99] if positivity_violations > 5 else None,
        "warnings": (
            ["Auto-finalized due to incomplete analysis"]
            if positivity_violations > 10 else []
        ),
        "reasoning": (
            f"Auto-finalized: {pct_overlap:.1f}% overlap, "
            f"{positivity_violations} positivity violations"
        ),
    }
