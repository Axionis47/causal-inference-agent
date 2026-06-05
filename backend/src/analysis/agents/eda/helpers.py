"""Pure helpers for the EDA agent.

These functions hold the loading and finalisation logic that previously lived
on the agent class. They take whatever state they need as arguments so they
can be unit-tested without instantiating the full ReAct stack.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from .output import EDAResult


def load_dataframe(dataframe_path: str | None) -> pd.DataFrame | None:
    """Read the parquet DataFrame the data_profiler wrote, or return None."""
    if dataframe_path and Path(dataframe_path).exists():
        return pd.read_parquet(dataframe_path)
    return None


def auto_finalize(
    missing_analysis: dict | None,
    outlier_results: dict[str, dict],
    vif_results: dict[str, float],
    balance_results: dict[str, dict],
) -> dict[str, Any]:
    """Heuristic fallback when the agent never calls finalize_eda."""
    score = 100.0
    issues: list[str] = []

    if missing_analysis and missing_analysis.get("has_missing"):
        missing_pct = missing_analysis.get("total_missing_pct", 0)
        if missing_pct > 30:
            score -= 30
            issues.append(f"High missing data: {missing_pct:.1f}%")
        elif missing_pct > 10:
            score -= 15
            issues.append(f"Moderate missing data: {missing_pct:.1f}%")
        elif missing_pct > 5:
            score -= 5
            issues.append(f"Some missing data: {missing_pct:.1f}%")

        if missing_analysis.get("differential_missing"):
            score -= 10
            issues.append("Missing data differs by treatment status")

    if outlier_results:
        outlier_cols = len(outlier_results)
        if outlier_cols > 5:
            score -= 15
            issues.append(f"Many columns with outliers: {outlier_cols}")
        elif outlier_cols > 2:
            score -= 8
            issues.append(f"Some columns with outliers: {outlier_cols}")

    if vif_results:
        severe = sum(1 for v in vif_results.values() if v > 10)
        moderate = sum(1 for v in vif_results.values() if 5 < v <= 10)
        if severe > 0:
            score -= 15
            issues.append(f"Severe multicollinearity in {severe} variables")
        elif moderate > 0:
            score -= 8
            issues.append(f"Moderate multicollinearity in {moderate} variables")

    if balance_results:
        imbalanced = sum(1 for v in balance_results.values() if not v.get("is_balanced", True))
        if imbalanced > 5:
            score -= 15
            issues.append(f"Many imbalanced covariates: {imbalanced}")
        elif imbalanced > 2:
            score -= 8
            issues.append(f"Some imbalanced covariates: {imbalanced}")
        elif imbalanced > 0:
            score -= 5
            issues.append(f"Imbalanced covariates: {imbalanced}")

    score = max(0, score)

    key_findings: list[str] = []
    if score >= 80:
        key_findings.append("Data quality is good for causal analysis")
    elif score >= 60:
        key_findings.append("Data quality is moderate - some issues need attention")
    else:
        key_findings.append("Significant data quality concerns identified")
    key_findings.extend(issues[:5])

    recommendations: list[str] = []
    if outlier_results:
        recommendations.append("Consider robust estimation methods or outlier treatment")
    if vif_results and any(v > 5 for v in vif_results.values()):
        recommendations.append("Address multicollinearity through variable selection or regularization")
    if balance_results and any(not v.get("is_balanced", True) for v in balance_results.values()):
        recommendations.append("Use propensity score methods to address covariate imbalance")
    if not recommendations:
        recommendations = ["Proceed with causal analysis", "Consider multiple estimation methods"]

    if score >= 80:
        causal_readiness = "ready"
    elif score >= 50:
        causal_readiness = "needs_attention"
    else:
        causal_readiness = "significant_concerns"

    return {
        "data_quality_score": score,
        "key_findings": key_findings,
        "data_quality_issues": issues,
        "recommendations": recommendations,
        "causal_readiness": causal_readiness,
    }


def populate_eda_result(
    eda_result: EDAResult,
    final_result: dict[str, Any],
    analyzed_distributions: dict[str, dict],
    outlier_results: dict[str, dict],
    vif_results: dict[str, float],
    balance_results: dict[str, dict],
) -> None:
    """Fold finalize output and collected evidence into the EDAResult slot."""
    eda_result.data_quality_score = final_result.get("data_quality_score", 50.0)
    eda_result.data_quality_issues = final_result.get("data_quality_issues", [])
    eda_result.summary_table = {
        "key_findings": final_result.get("key_findings", []),
        "recommendations": final_result.get("recommendations", []),
        "causal_readiness": final_result.get("causal_readiness", "needs_attention"),
    }
    eda_result.distribution_stats = analyzed_distributions
    eda_result.outliers = outlier_results
    if not eda_result.vif_scores:
        eda_result.vif_scores = vif_results
    if not eda_result.covariate_balance:
        eda_result.covariate_balance = balance_results
    eda_result.plot_captions = final_result.get("plot_captions", {}) or {}


def initial_observation_text(state) -> str:
    """Lean initial observation that just frames the dataset and workflow."""
    dataset_name = state.dataset_info.name or "dataset"
    n_samples = state.dataset_info.n_samples or "unknown"
    n_features = state.dataset_info.n_features or "unknown"
    return f"""Perform exploratory data analysis for causal inference.
Dataset: {dataset_name}
Size: {n_samples} samples x {n_features} features

Start by:
1. Querying domain knowledge for treatment/outcome hints
2. Getting data overview
3. Analyzing key variables (treatment, outcome)
4. Checking covariate balance

Finalize when you have assessed data quality for causal analysis."""
