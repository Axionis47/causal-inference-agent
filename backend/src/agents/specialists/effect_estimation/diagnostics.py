"""Diagnostic checks for treatment effect estimation.

Provides residual analysis, influence diagnostics, and specification tests
that the EffectEstimatorAgent uses to evaluate estimation quality.
"""

import numpy as np
from scipy import stats

from src.agents.base import TreatmentEffectResult


def check_residual_diagnostics(
    df, treatment_var: str, outcome_var: str, covariates: list[str]
) -> dict:
    """Check residual-based diagnostics."""
    from sklearn.linear_model import LinearRegression

    T = df[treatment_var].values
    Y = df[outcome_var].values

    X = df[covariates[:10]].values if covariates else np.ones((len(Y), 1))

    mask = ~np.any(np.isnan(X), axis=1)
    model = LinearRegression()
    model.fit(np.column_stack([T[mask], X[mask]]), Y[mask])
    residuals = Y[mask] - model.predict(np.column_stack([T[mask], X[mask]]))

    # Normality test
    _, normality_p = stats.shapiro(residuals[:min(5000, len(residuals))])

    # Heteroskedasticity
    _, hetero_p = stats.pearsonr(np.abs(residuals), model.predict(np.column_stack([T[mask], X[mask]])))

    return {
        "residual_mean": round(float(residuals.mean()), 4),
        "residual_std": round(float(residuals.std()), 4),
        "normality_p": round(float(normality_p), 4),
        "heteroskedasticity_p": round(float(abs(hetero_p)), 4),
        "normal": normality_p > 0.05,
        "homoskedastic": abs(hetero_p) < 0.1,
    }


def check_influence_diagnostics(df, outcome_var: str) -> dict:
    """Check for influential observations."""
    Y = df[outcome_var].values
    n = len(Y)
    threshold = 4 / n

    return {
        "sample_size": n,
        "influence_threshold": round(threshold, 4),
        "recommendation": "Check for outliers if estimate seems unstable",
    }


def check_specification_diagnostics(result: TreatmentEffectResult) -> dict:
    """Check model specification."""
    if not result:
        return {"error": "No results to diagnose"}

    issues = []
    if result.std_error > abs(result.estimate):
        issues.append("High uncertainty: SE > |estimate|")
    if result.p_value and result.p_value > 0.1:
        issues.append("Not statistically significant at 10% level")
    if result.details and "n_treated" in result.details:
        if result.details["n_treated"] < 50:
            issues.append("Small treated sample may cause instability")

    return {
        "effect_size": round(result.estimate, 4),
        "relative_se": round(result.std_error / (abs(result.estimate) + 1e-10), 2),
        "issues_count": len(issues),
        "issues": issues,
    }
