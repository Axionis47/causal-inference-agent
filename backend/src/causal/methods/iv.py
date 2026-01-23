"""Instrumental Variables (IV/2SLS) Method."""

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

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
            first_stage_X = np.column_stack([np.ones(len(T)), Z, X_exog])
        else:
            first_stage_X = np.column_stack([np.ones(len(T)), Z])

        self._first_stage = sm.OLS(T, first_stage_X).fit()
        T_hat = self._first_stage.fittedvalues

        # Check first-stage F-statistic (instrument strength)
        # F-stat for excluded instruments
        n_instruments = Z.shape[1] if len(Z.shape) > 1 else 1
        self._first_stage_f = self._first_stage.fvalue

        # Second stage: Y ~ T_hat + X
        if X_exog is not None:
            second_stage_X = np.column_stack([np.ones(len(Y)), T_hat, X_exog])
        else:
            second_stage_X = np.column_stack([np.ones(len(Y)), T_hat])

        self._second_stage = sm.OLS(Y, second_stage_X).fit()

        # Correct standard errors for 2SLS
        # Use residuals from second stage with original T
        if X_exog is not None:
            original_X = np.column_stack([np.ones(len(Y)), T, X_exog])
        else:
            original_X = np.column_stack([np.ones(len(Y)), T])

        residuals = Y - self._second_stage.predict(
            np.column_stack([np.ones(len(Y)), T_hat, X_exog]) if X_exog is not None
            else np.column_stack([np.ones(len(Y)), T_hat])
        )
        sigma_sq = np.sum(residuals**2) / (len(Y) - second_stage_X.shape[1])

        # Correct variance-covariance matrix
        XtX_inv = np.linalg.inv(second_stage_X.T @ second_stage_X)
        self._corrected_se = np.sqrt(sigma_sq * np.diag(XtX_inv))

        self._n_obs = len(Y)
        self._n_instruments = n_instruments
        self._instruments = instruments

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
        diagnostics = {
            "first_stage_f": float(self._first_stage_f) if self._first_stage_f else None,
            "first_stage_r2": float(self._first_stage.rsquared),
            "second_stage_r2": float(self._second_stage.rsquared),
            "n_instruments": self._n_instruments,
            "weak_instrument": self._first_stage_f < 10 if self._first_stage_f else True,
        }

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
                "first_stage_f_critical": 10,  # Stock-Yogo weak instrument threshold
            },
        )

        return self._result

    def validate_assumptions(
        self, df: pd.DataFrame, treatment_col: str, outcome_col: str
    ) -> list[str]:
        """Validate IV assumptions."""
        violations = super().validate_assumptions(df, treatment_col, outcome_col)

        if self._fitted:
            # Check for weak instruments
            if self._first_stage_f and self._first_stage_f < 10:
                violations.append(
                    f"Weak instrument: First-stage F = {self._first_stage_f:.2f} < 10"
                )

            # Check first-stage significance
            for inst in self._instruments:
                idx = self._instruments.index(inst) + 1
                if self._first_stage.pvalues[idx] > 0.05:
                    violations.append(
                        f"Instrument {inst} not significant in first stage "
                        f"(p = {self._first_stage.pvalues[idx]:.4f})"
                    )

        return violations
