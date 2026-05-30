"""Pure helpers for the ReAct effect estimator.

DataFrame loading, the initial observation string, the OLS auto-finalize
fallback, and the covariate priority chain. Each helper takes exactly
what it needs so it can be unit-tested independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis.agents.base import AnalysisState, TreatmentEffectResult


def load_dataframe(dataframe_path: str | None) -> pd.DataFrame | None:
    """Load the parquet at dataframe_path, returning None when missing."""
    if dataframe_path and Path(dataframe_path).exists():
        return pd.read_parquet(dataframe_path)
    return None


def build_initial_observation(
    state: AnalysisState,
    df: pd.DataFrame | None,
    load_error: str | None = None,
) -> str:
    """Lean initial observation string.

    Reduces token usage by ~500 tokens per call: no full data profile
    dump up-front. The agent's tools pull specific info on demand.
    """
    obs = f"""Task: Estimate treatment effects for job {state.job_id}
Dataset: {state.dataset_info.name or state.dataset_info.url}

Key variables:
- Treatment: {state.treatment_variable or "Use get_treatment_outcome tool to confirm"}
- Outcome: {state.outcome_variable or "Use get_treatment_outcome tool to confirm"}
"""
    if state.data_profile:
        obs += f"- Samples: {state.data_profile.n_samples}, Features: {state.data_profile.n_features}\n"

    obs += """
Start by inspecting the data overview to understand what you're working with.
Then analyze the treatment variable, check assumptions, and run estimation methods."""

    if df is not None:
        obs += f"\n\nData loaded: {len(df)} rows, {len(df.columns)} columns"
    elif load_error is not None:
        obs += f"\n\nWarning: Could not load data: {load_error}"

    return obs


def resolve_baseline_covariates(
    df: pd.DataFrame | None,
    state: AnalysisState,
) -> list[str]:
    """Same priority chain as the imperative estimator, simplified.

    Skips the one-hot-encoding shortcut: the engine's run_method_safe
    already filters to numeric columns, and at this point the pipeline
    has already done any pre-processing it was going to do.
    """
    if df is None:
        return []
    treatment = state.treatment_variable
    outcome = state.outcome_variable

    def _filter(cols: list[str]) -> list[str]:
        return [
            c for c in cols
            if c in df.columns and c != treatment and c != outcome
        ]

    if state.proposed_dag and state.proposed_dag.adjustment_set:
        covs = _filter(state.proposed_dag.adjustment_set)
        if covs:
            return covs

    if state.confounder_discovery:
        ranked = state.confounder_discovery.get("ranked_confounders", [])
        if ranked and isinstance(ranked[0], dict):
            ranked = [c.get("variable") or c.get("name") or "" for c in ranked]
        covs = _filter([c for c in ranked if c])
        if covs:
            return covs

    if state.data_profile and state.data_profile.potential_confounders:
        covs = _filter(state.data_profile.potential_confounders)
        if covs:
            return covs

    return []


def auto_finalize_ols(
    df: pd.DataFrame | None,
    state: AnalysisState,
    results: list[TreatmentEffectResult],
    logger: Any,
    max_steps: int,
) -> pd.DataFrame | None:
    """Run an OLS baseline when the loop ended with no valid estimates.

    Resolves covariates from the same priority chain the imperative
    estimator uses (DAG adjustment set, confounder discovery, data
    profile potential confounders) and delegates to the
    EffectEstimatorEngine so encoding and sample-size gating runs
    identically to a normal run_method call. Updates results and
    state.treatment_effects in place; logs and swallows any exception
    so the surrounding orchestrator loop can continue.

    Returns the (possibly newly loaded) dataframe so the caller can
    keep it for any later use.
    """
    if df is None and state.dataframe_path:
        try:
            df = pd.read_parquet(state.dataframe_path)
        except Exception as exc:
            logger.warning("auto_finalize_load_failed", error=str(exc))
            return df

    if df is None:
        return df

    treatment = state.treatment_variable
    outcome = state.outcome_variable
    if not treatment or not outcome:
        return df

    covariates = resolve_baseline_covariates(df, state)

    try:
        from src.causal.estimators.effect_estimator import EffectEstimatorEngine

        engine = EffectEstimatorEngine(confidence_level=0.95)
        ols_result = engine.run_method_safe(
            method="ols",
            df=df,
            treatment_col=treatment,
            outcome_col=outcome,
            covariates=covariates,
            current_state=state,
        )
    except Exception as exc:
        logger.warning("auto_finalize_ols_raised", error=str(exc))
        state.push_decision(
            agent="effect_estimator_react",
            decision_type="auto_finalize_failed",
            choice="ols",
            reason=f"Fallback OLS raised: {exc}",
        )
        return df

    if ols_result is None:
        logger.warning("auto_finalize_ols_no_result")
        state.push_decision(
            agent="effect_estimator_react",
            decision_type="auto_finalize_failed",
            choice="ols",
            reason="Fallback OLS returned no result (insufficient data).",
        )
        return df

    ols_result.treatment_variable = treatment
    ols_result.outcome_variable = outcome
    results.append(ols_result)
    state.treatment_effects.append(ols_result)
    state.push_decision(
        agent="effect_estimator_react",
        decision_type="auto_finalize_baseline",
        choice="ols",
        reason=(
            f"ReAct loop exited with no valid estimates after {max_steps} steps; "
            f"emitted OLS baseline with {len(covariates)} covariate(s)."
        ),
    )
    return df
