"""Instrumental Variables (IV/2SLS) Method."""

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .base import BaseCausalMethod, MethodResult


class InstrumentalVariablesMethod(BaseCausalMethod):
    """Two-Stage Least Squares (2SLS) / Instrumental Variables.

    Estimates Local Average Treatment Effect (LATE) using instrumental
    variables that affect treatment but not outcome directly.

    Assumptions:
    - Relevance: Instrument predicts treatment
    - Exclusion: Instrument affects outcome only through treatment
    - Independence: Instrument is as-good-as-randomly assigned
    - Monotonicity: No defiers (for LATE interpretation)
    """

    METHOD_NAME = "Instrumental Variables (2SLS)"
    ESTIMAND = "LATE"

    def __init__(self, confidence_level: float = 0.95):
        """Initialize IV method."""
        super().__init__(confidence_level)
        self._first_stage = None
        self._second_stage = None

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        instruments: list[str] | None = None,
        **kwargs: Any,
    ) -> "InstrumentalVariablesMethod":
        """Fit 2SLS model.

        Args:
            df: DataFrame with data
            treatment_col: Endogenous treatment variable
            outcome_col: Outcome variable
            covariates: Exogenous control variables
            instruments: Instrumental variables (excluded instruments)

        Returns:
            Self for chaining
        """
        if not instruments:
            raise ValueError("IV requires at least one instrument")

        # Prepare data
        all_cols = [treatment_col, outcome_col] + instruments
        if covariates:
            all_cols.extend(covariates)
        df_clean = df.dropna(subset=all_cols)

        Y = df_clean[outcome_col].values
        T = df_clean[treatment_col].values
        n = len(Y)

        # Get instruments
        Z = df_clean[instruments].select_dtypes(include=[np.number]).values

        # Get covariates (exogenous controls)
        if covariates:
            valid_covs = [c for c in covariates if c in df_clean.columns]
            X_exog = df_clean[valid_covs].select_dtypes(include=[np.number]).values
        else:
            X_exog = None

        # First stage: T ~ Z + X
        if X_exog is not None:
            first_stage_X = np.column_stack([np.ones(n), Z, X_exog])
        else:
            first_stage_X = np.column_stack([np.ones(n), Z])

        self._first_stage = sm.OLS(T, first_stage_X).fit()
        T_hat = self._first_stage.fittedvalues

        # --- Partial F-statistic for excluded instruments ---
        n_instruments = Z.shape[1] if len(Z.shape) > 1 else 1

        # Restricted model: T ~ X_exog only (no instruments)
        if X_exog is not None:
            restricted_X = np.column_stack([np.ones(n), X_exog])
        else:
            restricted_X = np.ones((n, 1))
        restricted_model = sm.OLS(T, restricted_X).fit()

        RSS_restricted = np.sum(restricted_model.resid ** 2)
        RSS_unrestricted = np.sum(self._first_stage.resid ** 2)
        k_unrestricted = first_stage_X.shape[1]

        self._first_stage_f = (
            ((RSS_restricted - RSS_unrestricted) / n_instruments)
            / (RSS_unrestricted / (n - k_unrestricted))
        )

        # Second stage: Y ~ T_hat + X
        if X_exog is not None:
            second_stage_X = np.column_stack([np.ones(n), T_hat, X_exog])
        else:
            second_stage_X = np.column_stack([np.ones(n), T_hat])

        self._second_stage = sm.OLS(Y, second_stage_X).fit()

        # --- Correct 2SLS standard errors ---
        # Use residuals computed from original T, not T_hat
        beta_2sls = self._second_stage.params
        if X_exog is not None:
            original_X = np.column_stack([np.ones(n), T, X_exog])
        else:
            original_X = np.column_stack([np.ones(n), T])

        residuals_2sls = Y - original_X @ beta_2sls
        sigma_sq = np.sum(residuals_2sls ** 2) / (n - second_stage_X.shape[1])

        # Sandwich variance: (Z'X)^{-1} Z' sigma^2 I Z (X'Z)^{-1}
        # Simplified: sigma^2 * (X_hat'X_hat)^{-1} where X_hat uses predicted T
        XhXh_inv = np.linalg.inv(second_stage_X.T @ second_stage_X)
        self._corrected_se = np.sqrt(sigma_sq * np.diag(XhXh_inv))

        # --- Hansen J-test (overidentification) ---
        self._hansen_j = None
        self._hansen_j_pvalue = None
        if n_instruments > 1:
            # Regress 2SLS residuals on all exogenous variables (instruments + covariates)
            if X_exog is not None:
                all_exog = np.column_stack([np.ones(n), Z, X_exog])
            else:
                all_exog = np.column_stack([np.ones(n), Z])
            j_reg = sm.OLS(residuals_2sls, all_exog).fit()
            self._hansen_j = float(n * j_reg.rsquared)
            self._hansen_j_pvalue = float(1 - stats.chi2.cdf(self._hansen_j, n_instruments - 1))

        # --- Durbin-Wu-Hausman endogeneity test ---
        # Add first-stage residuals to second-stage equation; test their significance
        first_stage_resid = self._first_stage.resid
        if X_exog is not None:
            dwh_X = np.column_stack([np.ones(n), T, X_exog, first_stage_resid])
        else:
            dwh_X = np.column_stack([np.ones(n), T, first_stage_resid])
        dwh_model = sm.OLS(Y, dwh_X).fit()
        # The coefficient on the first-stage residual is the last one
        self._dwh_statistic = float(dwh_model.tvalues[-1] ** 2)
        self._dwh_pvalue = float(dwh_model.pvalues[-1])

        self._n_obs = n
        self._n_instruments = n_instruments
        self._instruments = instruments
        self._residuals_2sls = residuals_2sls

        self._fitted = True
        return self

    def estimate(self) -> MethodResult:
        """Compute LATE estimate.

        Returns:
            MethodResult with estimate and statistics
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        # LATE is the coefficient on instrumented treatment (index 1)
        late = self._second_stage.params[1]
        se = self._corrected_se[1]

        ci_lower, ci_upper = self._compute_ci(late, se)
        p_value = self._compute_p_value(late, se)

        # Diagnostics
        weak_instrument_severe = (
            self._first_stage_f < 10 if self._first_stage_f else True
        )
        weak_instrument_moderate = (
            self._first_stage_f < 16.38 if self._first_stage_f else True
        )

        diagnostics = {
            "first_stage_f_partial": float(self._first_stage_f) if self._first_stage_f else None,
            "first_stage_r2": float(self._first_stage.rsquared),
            "second_stage_r2": float(self._second_stage.rsquared),
            "n_instruments": self._n_instruments,
            "weak_instrument_severe": weak_instrument_severe,
            "weak_instrument_moderate": weak_instrument_moderate,
            "endogeneity_dwh_statistic": self._dwh_statistic,
            "endogeneity_dwh_pvalue": self._dwh_pvalue,
            "treatment_endogenous": self._dwh_pvalue < 0.05,
        }

        # Hansen J-test (only for overidentified models)
        if self._hansen_j is not None:
            diagnostics["hansen_j_statistic"] = self._hansen_j
            diagnostics["hansen_j_pvalue"] = self._hansen_j_pvalue
            diagnostics["overid_rejected"] = self._hansen_j_pvalue < 0.05

        # First stage coefficients on instruments
        instrument_coeffs = {}
        for i, inst in enumerate(self._instruments):
            instrument_coeffs[inst] = {
                "coef": float(self._first_stage.params[i + 1]),
                "pvalue": float(self._first_stage.pvalues[i + 1]),
            }
        diagnostics["instrument_coefficients"] = instrument_coeffs

        self._result = MethodResult(
            method=self.METHOD_NAME,
            estimand=self.ESTIMAND,
            estimate=float(late),
            std_error=float(se),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(p_value),
            n_treated=self._n_obs,
            n_control=0,  # Not applicable for IV
            assumptions_tested=["relevance", "exclusion", "independence", "monotonicity"],
            diagnostics=diagnostics,
            details={
                "instruments": self._instruments,
                "first_stage_f_critical_moderate": 16.38,
                "first_stage_f_critical_severe": 10,
            },
        )

        return self._result

    def validate_assumptions(
        self, df: pd.DataFrame, treatment_col: str, outcome_col: str
    ) -> list[str]:
        """Validate IV assumptions."""
        violations = super().validate_assumptions(df, treatment_col, outcome_col)

        if self._fitted:
            # Check for severely weak instruments (F < 10)
            if self._first_stage_f and self._first_stage_f < 10:
                violations.append(
                    f"Severely weak instrument: Partial F = {self._first_stage_f:.2f} < 10"
                )
            elif self._first_stage_f and self._first_stage_f < 16.38:
                violations.append(
                    f"Weak instrument (Stock-Yogo 5% relative bias): "
                    f"Partial F = {self._first_stage_f:.2f} < 16.38"
                )

            # Check first-stage significance
            for inst in self._instruments:
                idx = self._instruments.index(inst) + 1
                if self._first_stage.pvalues[idx] > 0.05:
                    violations.append(
                        f"Instrument {inst} not significant in first stage "
                        f"(p = {self._first_stage.pvalues[idx]:.4f})"
                    )

            # Check Hansen J overidentification
            if self._hansen_j is not None and self._hansen_j_pvalue < 0.05:
                violations.append(
                    f"Hansen J overidentification test rejected "
                    f"(J = {self._hansen_j:.2f}, p = {self._hansen_j_pvalue:.4f}): "
                    f"instruments may not be valid"
                )

            # Check endogeneity via DWH test
            if self._dwh_pvalue > 0.05:
                violations.append(
                    f"DWH endogeneity test not significant "
                    f"(p = {self._dwh_pvalue:.4f}): treatment may not be endogenous, "
                    f"OLS may be preferred"
                )

        return violations
