"""Effect Estimator Engine - Unified interface for all causal methods."""

from typing import Any

import pandas as pd

from ..methods import (
    AIPWMethod,
    CausalForestMethod,
    DifferenceInDifferencesMethod,
    DoubleMLMethod,
    InstrumentalVariablesMethod,
    IPWMethod,
    MethodResult,
    OLSMethod,
    PSMMethod,
    RegressionDiscontinuityMethod,
    SLearner,
    TLearner,
    XLearner,
)


class EffectEstimatorEngine:
    """Unified engine for running causal effect estimation methods.

    Provides a clean interface for:
    1. Running individual methods
    2. Running multiple methods for robustness
    3. Selecting appropriate methods based on data characteristics
    """

    # Available methods and their classes
    METHODS = {
        "ols": OLSMethod,
        "psm": PSMMethod,
        "ipw": IPWMethod,
        "aipw": AIPWMethod,
        "did": DifferenceInDifferencesMethod,
        "iv": InstrumentalVariablesMethod,
        "rdd": RegressionDiscontinuityMethod,
        "s_learner": SLearner,
        "t_learner": TLearner,
        "x_learner": XLearner,
        "causal_forest": CausalForestMethod,
        "double_ml": DoubleMLMethod,
    }

    # Method requirements
    METHOD_REQUIREMENTS = {
        "ols": {"min_samples": 30, "requires_covariates": False},
        "psm": {"min_samples": 100, "requires_covariates": True},
        "ipw": {"min_samples": 100, "requires_covariates": True},
        "aipw": {"min_samples": 100, "requires_covariates": True},
        "did": {"min_samples": 50, "requires_panel": True},
        "iv": {"min_samples": 100, "requires_instruments": True},
        "rdd": {"min_samples": 100, "requires_running_var": True},
        "s_learner": {"min_samples": 200, "requires_covariates": True},
        "t_learner": {"min_samples": 200, "requires_covariates": True},
        "x_learner": {"min_samples": 200, "requires_covariates": True},
        "causal_forest": {"min_samples": 500, "requires_covariates": True},
        "double_ml": {"min_samples": 200, "requires_covariates": True},
    }

    def __init__(self, confidence_level: float = 0.95):
        """Initialize the engine.

        Args:
            confidence_level: Confidence level for all methods
        """
        self.confidence_level = confidence_level
        self._results: list[MethodResult] = []

    def run_method(
        self,
        method_name: str,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> MethodResult:
        """Run a single causal estimation method.

        Args:
            method_name: Name of method to run
            df: DataFrame with data
            treatment_col: Treatment variable name
            outcome_col: Outcome variable name
            covariates: Covariate column names
            **kwargs: Method-specific arguments

        Returns:
            MethodResult with estimates
        """
        method_name = method_name.lower().replace("-", "_").replace(" ", "_")

        if method_name not in self.METHODS:
            raise ValueError(f"Unknown method: {method_name}. Available: {list(self.METHODS.keys())}")

        # Get method class
        method_class = self.METHODS[method_name]

        # Initialize and fit
        method = method_class(confidence_level=self.confidence_level)
        method.fit(df, treatment_col, outcome_col, covariates, **kwargs)

        # Get result
        result = method.estimate()
        self._results.append(result)

        return result

    def run_multiple(
        self,
        methods: list[str],
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> list[MethodResult]:
        """Run multiple methods for robustness.

        Args:
            methods: List of method names
            df: DataFrame with data
            treatment_col: Treatment variable name
            outcome_col: Outcome variable name
            covariates: Covariate column names
            **kwargs: Additional arguments

        Returns:
            List of MethodResults
        """
        results = []

        for method_name in methods:
            try:
                result = self.run_method(
                    method_name, df, treatment_col, outcome_col, covariates, **kwargs
                )
                results.append(result)
            except Exception as e:
                # Log but continue with other methods
                print(f"Method {method_name} failed: {e}")

        return results

    def suggest_methods(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        has_panel: bool = False,
        has_instruments: list[str] | None = None,
        has_running_var: str | None = None,
    ) -> list[str]:
        """Suggest appropriate methods based on data characteristics.

        Args:
            df: DataFrame with data
            treatment_col: Treatment variable
            outcome_col: Outcome variable
            covariates: Available covariates
            has_panel: Whether data has panel structure
            has_instruments: Available instrumental variables
            has_running_var: Running variable for RDD

        Returns:
            List of recommended method names
        """
        n_samples = len(df)
        has_covariates = covariates and len(covariates) > 0

        recommended = []

        # Always recommend OLS as baseline
        if n_samples >= 30:
            recommended.append("ols")

        # Propensity methods if covariates available
        if has_covariates and n_samples >= 100:
            recommended.extend(["ipw", "aipw"])
            if n_samples >= 100:
                recommended.append("psm")

        # DiD if panel data
        if has_panel and n_samples >= 50:
            recommended.append("did")

        # IV if instruments available
        if has_instruments and n_samples >= 100:
            recommended.append("iv")

        # RDD if running variable available
        if has_running_var and n_samples >= 100:
            recommended.append("rdd")

        # ML methods for larger samples
        if has_covariates and n_samples >= 200:
            recommended.extend(["double_ml", "s_learner", "t_learner"])

            if n_samples >= 500:
                recommended.extend(["causal_forest", "x_learner"])

        return recommended

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all results.

        Returns:
            Summary statistics across methods
        """
        if not self._results:
            return {"n_methods": 0}

        estimates = [r.estimate for r in self._results]

        return {
            "n_methods": len(self._results),
            "methods": [r.method for r in self._results],
            "mean_estimate": float(sum(estimates) / len(estimates)),
            "median_estimate": float(sorted(estimates)[len(estimates) // 2]),
            "min_estimate": float(min(estimates)),
            "max_estimate": float(max(estimates)),
            "estimate_range": float(max(estimates) - min(estimates)),
            "all_positive": all(e > 0 for e in estimates),
            "all_negative": all(e < 0 for e in estimates),
            "all_significant": all(
                r.p_value is not None and r.p_value < 0.05 for r in self._results
            ),
        }

    @property
    def results(self) -> list[MethodResult]:
        """Get all results."""
        return self._results

    def clear_results(self):
        """Clear stored results."""
        self._results = []
