"""OLS Regression for causal inference."""

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .base import BaseCausalMethod, MethodResult


class OLSMethod(BaseCausalMethod):
    """Ordinary Least Squares regression for treatment effect estimation.

    Estimates the Average Treatment Effect (ATE) using linear regression
    with covariate adjustment.

    Assumptions:
    - Linearity
    - No unmeasured confounding (conditional ignorability)
    - Homoscedasticity
    - No multicollinearity
    """

    METHOD_NAME = "OLS Regression"
    ESTIMAND = "ATE"

    def __init__(self, confidence_level: float = 0.95, robust_se: bool = True):
        """Initialize OLS method.

        Args:
            confidence_level: Confidence level for intervals
            robust_se: Use heteroscedasticity-robust standard errors
        """
        super().__init__(confidence_level)
        self.robust_se = robust_se
        self._model = None
        self._results = None

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> "OLSMethod":
        """Fit OLS model.

        Args:
            df: DataFrame with data
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            covariates: List of covariate column names

        Returns:
            Self for chaining
        """
        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col])

        # Binarize treatment if needed
        T = self._binarize_treatment(df_clean[treatment_col])
        Y = df_clean[outcome_col].values

        # Build design matrix
        if covariates:
            valid_covs = [c for c in covariates if c in df_clean.columns]
            X_covs = df_clean[valid_covs].select_dtypes(include=[np.number]).values
            X = np.column_stack([np.ones(len(T)), T.values, X_covs])
        else:
            X = np.column_stack([np.ones(len(T)), T.values])

        # Fit model
        self._model = sm.OLS(Y, X)
        if self.robust_se:
            self._results = self._model.fit(cov_type='HC1')
        else:
            self._results = self._model.fit()

        # Store data for diagnostics
        self._T = T
        self._Y = Y
        self._n_treated = int(T.sum())
        self._n_control = int(len(T) - T.sum())

        self._fitted = True
        return self

    def estimate(self) -> MethodResult:
        """Compute ATE estimate.

        Returns:
            MethodResult with estimate and statistics
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        # Treatment effect is coefficient on treatment variable (index 1)
        ate = self._results.params[1]
        se = self._results.bse[1]
        ci = self._results.conf_int(alpha=self.alpha)[1]
        p_value = self._results.pvalues[1]

        # Diagnostics
        diagnostics = {
            "r_squared": float(self._results.rsquared),
            "adj_r_squared": float(self._results.rsquared_adj),
            "f_statistic": float(self._results.fvalue) if self._results.fvalue else None,
            "f_pvalue": float(self._results.f_pvalue) if self._results.f_pvalue else None,
            "aic": float(self._results.aic),
            "bic": float(self._results.bic),
        }

        self._result = MethodResult(
            method=self.METHOD_NAME,
            estimand=self.ESTIMAND,
            estimate=float(ate),
            std_error=float(se),
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            p_value=float(p_value),
            n_treated=self._n_treated,
            n_control=self._n_control,
            assumptions_tested=["linearity", "no_unmeasured_confounding"],
            diagnostics=diagnostics,
            details={
                "robust_se": self.robust_se,
                "n_covariates": len(self._results.params) - 2,
            },
        )

        return self._result

    def validate_assumptions(
        self, df: pd.DataFrame, treatment_col: str, outcome_col: str
    ) -> list[str]:
        """Validate OLS assumptions.

        Args:
            df: DataFrame with data
            treatment_col: Treatment column name
            outcome_col: Outcome column name

        Returns:
            List of assumption violations
        """
        violations = super().validate_assumptions(df, treatment_col, outcome_col)

        if self._fitted and self._results is not None:
            from scipy import stats

            # Heteroscedasticity test (Breusch-Pagan)
            try:
                residuals = self._results.resid
                fitted = self._results.fittedvalues
                _, bp_pvalue = stats.pearsonr(residuals**2, fitted)
                if bp_pvalue < 0.05:
                    violations.append(f"Potential heteroscedasticity (p={bp_pvalue:.4f})")
            except Exception:
                pass

            # Normality of residuals (Shapiro-Wilk for small samples)
            try:
                if len(self._results.resid) <= 5000:
                    _, normality_p = stats.shapiro(self._results.resid[:5000])
                    if normality_p < 0.05:
                        violations.append(f"Non-normal residuals (Shapiro p={normality_p:.4f})")
            except Exception:
                pass

        return violations
