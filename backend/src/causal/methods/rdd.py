"""Regression Discontinuity Design (RDD) Method."""

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .base import BaseCausalMethod, MethodResult


class RegressionDiscontinuityMethod(BaseCausalMethod):
    """Regression Discontinuity Design (RDD).

    Estimates Local Average Treatment Effect at the cutoff using
    discontinuity in treatment assignment.

    Supports:
    - Sharp RDD: Treatment deterministically assigned at cutoff
    - Fuzzy RDD: Treatment probability jumps at cutoff (uses 2SLS)

    Assumptions:
    - Continuity of potential outcomes at cutoff
    - No manipulation of running variable
    - Local randomization around cutoff
    """

    METHOD_NAME = "Regression Discontinuity"
    ESTIMAND = "LATE"

    def __init__(
        self,
        confidence_level: float = 0.95,
        bandwidth: float | None = None,
        polynomial_order: int = 1,
        fuzzy: bool = False,
    ):
        """Initialize RDD.

        Args:
            confidence_level: Confidence level for intervals
            bandwidth: Bandwidth around cutoff (None for automatic)
            polynomial_order: Order of polynomial for local regression
            fuzzy: Whether to use fuzzy RDD (2SLS)
        """
        super().__init__(confidence_level)
        self.bandwidth = bandwidth
        self.polynomial_order = polynomial_order
        self.fuzzy = fuzzy

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        running_var: str | None = None,
        cutoff: float = 0,
        **kwargs: Any,
    ) -> "RegressionDiscontinuityMethod":
        """Fit RDD model.

        Args:
            df: DataFrame with data
            treatment_col: Treatment variable
            outcome_col: Outcome variable
            covariates: Additional controls
            running_var: Running variable that determines treatment
            cutoff: Cutoff value for treatment assignment

        Returns:
            Self for chaining
        """
        if not running_var:
            raise ValueError("RDD requires a running variable")

        # Prepare data
        all_cols = [treatment_col, outcome_col, running_var]
        if covariates:
            all_cols.extend(covariates)
        df_clean = df.dropna(subset=all_cols)

        Y = df_clean[outcome_col].values
        T = self._binarize_treatment(df_clean[treatment_col]).values
        R = df_clean[running_var].values

        # Center running variable at cutoff
        R_centered = R - cutoff

        # Determine bandwidth
        if self.bandwidth is None:
            # Simple rule-of-thumb bandwidth (IK optimal would be better)
            self.bandwidth = 1.5 * np.std(R_centered) * (len(R) ** (-1/5))

        # Filter to observations within bandwidth
        in_bandwidth = np.abs(R_centered) <= self.bandwidth
        Y_bw = Y[in_bandwidth]
        T_bw = T[in_bandwidth]
        R_bw = R_centered[in_bandwidth]

        # Create above-cutoff indicator
        above = (R_bw >= 0).astype(float)

        # Build design matrix with polynomial terms
        X_list = [np.ones(len(Y_bw)), above, R_bw, above * R_bw]

        # Add higher-order polynomial terms if requested
        for p in range(2, self.polynomial_order + 1):
            X_list.append(R_bw ** p)
            X_list.append(above * (R_bw ** p))

        X = np.column_stack(X_list)

        if self.fuzzy:
            # Fuzzy RDD: Use 2SLS with above-cutoff as instrument for treatment
            # First stage: T ~ above + f(R)
            first_stage = sm.OLS(T_bw, X).fit()
            T_hat = first_stage.fittedvalues

            # Replace above with T_hat
            X_second = X.copy()
            X_second[:, 1] = T_hat

            self._model = sm.OLS(Y_bw, X_second)
            self._results = self._model.fit()

            self._first_stage_coef = first_stage.params[1]
            self._first_stage_f = first_stage.fvalue
        else:
            # Sharp RDD: Direct regression
            self._model = sm.OLS(Y_bw, X)
            self._results = self._model.fit(cov_type='HC1')

        # Store counts
        self._n_below = int((R_bw < 0).sum())
        self._n_above = int((R_bw >= 0).sum())
        self._n_total = len(Y_bw)
        self._cutoff = cutoff
        self._running_var = running_var

        # For diagnostic plots
        self._R_bw = R_bw
        self._Y_bw = Y_bw
        self._above = above

        self._fitted = True
        return self

    def estimate(self) -> MethodResult:
        """Compute RDD estimate at the cutoff.

        Returns:
            MethodResult with estimate and statistics
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        # Treatment effect at cutoff is coefficient on 'above' indicator (index 1)
        rdd_estimate = self._results.params[1]
        se = self._results.bse[1]
        ci = self._results.conf_int(alpha=self.alpha)[1]
        p_value = self._results.pvalues[1]

        # Compute local means
        mean_below = float(np.mean(self._Y_bw[self._above == 0]))
        mean_above = float(np.mean(self._Y_bw[self._above == 1]))

        diagnostics = {
            "bandwidth": float(self.bandwidth),
            "n_below_cutoff": self._n_below,
            "n_above_cutoff": self._n_above,
            "mean_below": mean_below,
            "mean_above": mean_above,
            "raw_jump": mean_above - mean_below,
            "r_squared": float(self._results.rsquared),
            "polynomial_order": self.polynomial_order,
        }

        if self.fuzzy:
            diagnostics["first_stage_coef"] = float(self._first_stage_coef)
            diagnostics["first_stage_f"] = float(self._first_stage_f)
            diagnostics["fuzzy"] = True
        else:
            diagnostics["fuzzy"] = False

        self._result = MethodResult(
            method=self.METHOD_NAME,
            estimand=self.ESTIMAND,
            estimate=float(rdd_estimate),
            std_error=float(se),
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            p_value=float(p_value),
            n_treated=self._n_above,
            n_control=self._n_below,
            assumptions_tested=["continuity", "no_manipulation", "local_randomization"],
            diagnostics=diagnostics,
            details={
                "cutoff": self._cutoff,
                "running_variable": self._running_var,
                "bandwidth": self.bandwidth,
            },
        )

        return self._result

    def validate_assumptions(
        self, df: pd.DataFrame, treatment_col: str, outcome_col: str
    ) -> list[str]:
        """Validate RDD assumptions."""
        violations = super().validate_assumptions(df, treatment_col, outcome_col)

        if self._fitted:
            # Check for sufficient observations on each side
            if self._n_below < 20 or self._n_above < 20:
                violations.append(
                    f"Few observations near cutoff: {self._n_below} below, {self._n_above} above"
                )

            # McCrary density test (simplified version)
            # Check if density of running variable is continuous at cutoff
            density_below = self._n_below / self._n_total
            density_above = self._n_above / self._n_total

            if abs(density_below - density_above) > 0.3:
                violations.append(
                    f"Possible manipulation: density ratio = {density_above/density_below:.2f}"
                )

            # For fuzzy RDD, check first stage
            if self.fuzzy and hasattr(self, '_first_stage_f'):
                if self._first_stage_f < 10:
                    violations.append(
                        f"Weak first stage in fuzzy RDD: F = {self._first_stage_f:.2f}"
                    )

        return violations
