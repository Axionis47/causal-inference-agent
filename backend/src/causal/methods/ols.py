"""OLS Regression for causal inference."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .base import BaseCausalMethod, MethodResult

logger = logging.getLogger(__name__)


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
            cov_df = df_clean[valid_covs]
            numeric_cov_df = cov_df.select_dtypes(include=[np.number])

            # Warn about dropped non-numeric covariates
            dropped_covs = set(valid_covs) - set(numeric_cov_df.columns)
            if dropped_covs:
                logger.warning(
                    "Dropped non-numeric covariates from OLS model: %s. "
                    "Consider encoding categorical variables before fitting.",
                    sorted(dropped_covs),
                )

            X_covs = numeric_cov_df.values
            X = np.column_stack([np.ones(len(T)), T.values, X_covs])
            self._covariate_names = list(numeric_cov_df.columns)
        else:
            X = np.column_stack([np.ones(len(T)), T.values])
            self._covariate_names = []

        # Store the design matrix for diagnostics
        self._X = X

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

        # VIF for covariates (columns index >= 2 are covariates; 0=const, 1=treatment)
        if self._X.shape[1] > 2:
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor

                vif_values = {}
                high_vif_covs = []
                for i, cov_name in enumerate(self._covariate_names):
                    col_idx = i + 2  # offset for const and treatment
                    vif_val = variance_inflation_factor(self._X, col_idx)
                    vif_values[cov_name] = float(vif_val)
                    if vif_val > 10:
                        high_vif_covs.append((cov_name, vif_val))

                diagnostics["vif"] = vif_values
                if high_vif_covs:
                    logger.warning(
                        "High multicollinearity detected (VIF > 10): %s",
                        {name: f"{val:.1f}" for name, val in high_vif_covs},
                    )
                    diagnostics["vif_warning"] = (
                        f"Covariates with VIF > 10: "
                        f"{', '.join(f'{n} ({v:.1f})' for n, v in high_vif_covs)}"
                    )
            except Exception as e:
                logger.debug("Could not compute VIF: %s", e)

        # RESET test for functional form misspecification
        try:
            from statsmodels.stats.diagnostic import linear_reset

            reset_result = linear_reset(self._results, power=3, use_f=True)
            diagnostics["reset_f_stat"] = float(reset_result.fvalue)
            diagnostics["reset_f_pval"] = float(reset_result.pvalue)
            if reset_result.pvalue < 0.05:
                diagnostics["reset_warning"] = (
                    f"RESET test rejects linear specification (p={reset_result.pvalue:.4f}). "
                    "Consider adding polynomial terms or using a nonlinear model."
                )
        except ImportError:
            logger.debug("linear_reset not available in this statsmodels version")
        except Exception as e:
            logger.debug("RESET test failed: %s", e)

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

            residuals = self._results.resid
            n = len(residuals)

            # Heteroscedasticity test (Breusch-Pagan using proper LM test)
            try:
                from statsmodels.stats.diagnostic import het_breuschpagan

                lm_stat, lm_pval, f_stat, f_pval = het_breuschpagan(
                    residuals, self._X
                )
                if lm_pval < 0.05:
                    violations.append(
                        f"Heteroscedasticity detected (Breusch-Pagan LM={lm_stat:.2f}, "
                        f"p={lm_pval:.4f})"
                    )
            except Exception as e:
                logger.debug("Breusch-Pagan test failed: %s", e)

            # Normality of residuals
            # Use Shapiro-Wilk for n <= 2000, Anderson-Darling for n > 2000
            try:
                if n <= 2000:
                    _, normality_p = stats.shapiro(residuals)
                    if normality_p < 0.05:
                        violations.append(
                            f"Non-normal residuals (Shapiro-Wilk p={normality_p:.4f})"
                        )
                else:
                    ad_result = stats.anderson(residuals, dist='norm')
                    # Anderson-Darling: compare statistic to 5% critical value (index 2)
                    if ad_result.statistic > ad_result.critical_values[2]:
                        violations.append(
                            f"Non-normal residuals (Anderson-Darling stat="
                            f"{ad_result.statistic:.4f}, 5% critical="
                            f"{ad_result.critical_values[2]:.4f})"
                        )
            except Exception as e:
                logger.debug("Normality test failed: %s", e)

        return violations
