"""Regression Discontinuity Design (RDD) Method."""

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

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

        # Store full data for covariate balance tests
        self._df_clean = df_clean
        self._covariates = covariates
        self._treatment_col = treatment_col
        self._outcome_col = outcome_col

        # --- IK-style optimal bandwidth using IQR ---
        if self.bandwidth is None:
            iqr = np.percentile(R_centered, 75) - np.percentile(R_centered, 25)
            # Guard against zero IQR (e.g., discrete running variable)
            if iqr < 1e-10:
                iqr = np.std(R_centered)
            bandwidth = 2.5 * iqr * (len(R_centered) ** (-1 / 5))

            # Ensure minimum bandwidth captures at least 10% of observations per side
            sorted_below = np.sort(np.abs(R_centered[R_centered < 0]))
            sorted_above = np.sort(R_centered[R_centered >= 0])
            min_n_per_side = max(int(0.1 * len(R_centered[R_centered < 0])), 5)
            if len(sorted_below) > min_n_per_side:
                min_bw_below = sorted_below[min_n_per_side - 1]
            else:
                min_bw_below = sorted_below[-1] if len(sorted_below) > 0 else bandwidth
            if len(sorted_above) > min_n_per_side:
                min_bw_above = sorted_above[min_n_per_side - 1]
            else:
                min_bw_above = sorted_above[-1] if len(sorted_above) > 0 else bandwidth
            min_bw = max(min_bw_below, min_bw_above)
            self.bandwidth = max(bandwidth, min_bw)

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

        # Store counts and data for diagnostics
        self._n_below = int((R_bw < 0).sum())
        self._n_above = int((R_bw >= 0).sum())
        self._n_total = len(Y_bw)
        self._cutoff = cutoff
        self._running_var = running_var

        # For diagnostic plots and bandwidth sensitivity
        self._R_bw = R_bw
        self._Y_bw = Y_bw
        self._above = above
        self._R_centered_full = R_centered
        self._Y_full = Y
        self._T_full = T
        self._in_bandwidth = in_bandwidth

        self._fitted = True
        return self

    def _fit_at_bandwidth(self, R_centered, Y, T, bw):
        """Fit RDD at a specific bandwidth and return estimate and SE."""
        in_bw = np.abs(R_centered) <= bw
        Y_bw = Y[in_bw]
        R_bw = R_centered[in_bw]
        above = (R_bw >= 0).astype(float)

        if np.sum(R_bw < 0) < 5 or np.sum(R_bw >= 0) < 5:
            return None, None, int(in_bw.sum())

        X_list = [np.ones(len(Y_bw)), above, R_bw, above * R_bw]
        for p in range(2, self.polynomial_order + 1):
            X_list.append(R_bw ** p)
            X_list.append(above * (R_bw ** p))
        X = np.column_stack(X_list)

        try:
            results = sm.OLS(Y_bw, X).fit(cov_type='HC1')
            return float(results.params[1]), float(results.bse[1]), int(in_bw.sum())
        except Exception:
            return None, None, int(in_bw.sum())

    def _mccrary_density_test(self, R_centered):
        """McCrary density test using local polynomial density estimation.

        Bins the running variable near the cutoff, fits local polynomial
        separately on each side, and tests for a discontinuity in density.

        Returns:
            Tuple of (test_statistic, p_value, log_density_ratio)
        """
        # Create bins near cutoff
        n = len(R_centered)
        n_bins = max(int(np.sqrt(n)), 20)

        # Use data within 2x bandwidth for density estimation
        bw_density = 2 * self.bandwidth
        near_cutoff = R_centered[(np.abs(R_centered) <= bw_density)]

        if len(near_cutoff) < 20:
            return 0.0, 1.0, 0.0

        # Bin the data
        bins = np.linspace(-bw_density, bw_density, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_width = bins[1] - bins[0]
        counts, _ = np.histogram(R_centered, bins=bins)

        # Normalize to density
        density = counts / (len(R_centered) * bin_width)

        # Separate bins below and above cutoff
        below_mask = bin_centers < 0
        above_mask = bin_centers >= 0

        if np.sum(below_mask) < 3 or np.sum(above_mask) < 3:
            return 0.0, 1.0, 0.0

        # Fit local polynomial (quadratic) on each side using kernel weights
        def fit_local_poly(centers, dens, target=0.0):
            """Fit local polynomial and predict density at target."""
            h = bw_density / 2
            weights = np.exp(-0.5 * ((centers - target) / h) ** 2)
            X = np.column_stack([np.ones(len(centers)), centers, centers ** 2])
            W = np.diag(weights)
            try:
                beta = np.linalg.lstsq(W @ X, W @ dens, rcond=None)[0]
                pred = beta[0] + beta[1] * target + beta[2] * target ** 2
                # SE via delta method
                XWX_inv = np.linalg.inv(X.T @ W @ X)
                residuals = dens - X @ beta
                sigma2 = np.sum(weights * residuals ** 2) / max(len(centers) - 3, 1)
                se = np.sqrt(sigma2 * XWX_inv[0, 0])
                return pred, se
            except (np.linalg.LinAlgError, ValueError):
                return np.mean(dens), np.std(dens) / np.sqrt(len(dens))

        pred_below, se_below = fit_local_poly(
            bin_centers[below_mask], density[below_mask], target=0.0
        )
        pred_above, se_above = fit_local_poly(
            bin_centers[above_mask], density[above_mask], target=0.0
        )

        # Test statistic for discontinuity
        diff = pred_above - pred_below
        se_diff = np.sqrt(se_below ** 2 + se_above ** 2)

        if se_diff < 1e-10:
            return 0.0, 1.0, 0.0

        t_stat = diff / se_diff
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        # Log density ratio
        if pred_below > 0 and pred_above > 0:
            log_ratio = float(np.log(pred_above / pred_below))
        else:
            log_ratio = 0.0

        return float(t_stat), float(p_value), log_ratio

    def _covariate_balance_test(self, df_clean, covariates, running_var, cutoff):
        """Test covariate balance at the cutoff.

        For each covariate, run the RDD specification with the covariate
        as outcome. Significant effects indicate imbalance.

        Returns:
            Dict mapping covariate name to (estimate, p_value, balanced).
        """
        results = {}
        if not covariates:
            return results

        R_centered = df_clean[running_var].values - cutoff
        in_bw = np.abs(R_centered) <= self.bandwidth
        R_bw = R_centered[in_bw]
        above = (R_bw >= 0).astype(float)

        if np.sum(R_bw < 0) < 5 or np.sum(R_bw >= 0) < 5:
            return results

        for cov in covariates:
            if cov not in df_clean.columns:
                continue
            cov_vals = df_clean[cov].values
            if not np.issubdtype(cov_vals.dtype, np.number):
                continue
            C_bw = cov_vals[in_bw]
            if np.std(C_bw) < 1e-10:
                continue

            X_list = [np.ones(len(C_bw)), above, R_bw, above * R_bw]
            for p in range(2, self.polynomial_order + 1):
                X_list.append(R_bw ** p)
                X_list.append(above * (R_bw ** p))
            X = np.column_stack(X_list)

            try:
                cov_model = sm.OLS(C_bw, X).fit(cov_type='HC1')
                est = float(cov_model.params[1])
                pval = float(cov_model.pvalues[1])
                results[cov] = {
                    "estimate": est,
                    "p_value": pval,
                    "balanced": pval > 0.05,
                }
            except Exception:
                continue

        return results

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

        # --- McCrary density test ---
        mccrary_stat, mccrary_pval, log_ratio = self._mccrary_density_test(
            self._R_centered_full
        )
        diagnostics["mccrary_test_statistic"] = mccrary_stat
        diagnostics["mccrary_p_value"] = mccrary_pval
        diagnostics["mccrary_log_density_ratio"] = log_ratio
        diagnostics["manipulation_detected"] = mccrary_pval < 0.05

        # --- Covariate balance test ---
        cov_balance = self._covariate_balance_test(
            self._df_clean, self._covariates, self._running_var, self._cutoff
        )
        if cov_balance:
            diagnostics["covariate_balance"] = cov_balance
            n_imbalanced = sum(
                1 for v in cov_balance.values() if not v["balanced"]
            )
            diagnostics["n_covariates_imbalanced"] = n_imbalanced

        # --- Bandwidth sensitivity analysis ---
        h = self.bandwidth
        bw_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        sensitivity = {}
        for mult in bw_multipliers:
            bw_test = h * mult
            est, est_se, est_n = self._fit_at_bandwidth(
                self._R_centered_full, self._Y_full, self._T_full, bw_test
            )
            sensitivity[f"{mult:.2f}h"] = {
                "bandwidth": float(bw_test),
                "estimate": est,
                "std_error": est_se,
                "n_obs": est_n,
            }
        diagnostics["bandwidth_sensitivity"] = sensitivity

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

            # McCrary density test
            mccrary_stat, mccrary_pval, _ = self._mccrary_density_test(
                self._R_centered_full
            )
            if mccrary_pval < 0.05:
                violations.append(
                    f"McCrary density test detects possible manipulation "
                    f"(t = {mccrary_stat:.2f}, p = {mccrary_pval:.4f})"
                )

            # Covariate balance
            cov_balance = self._covariate_balance_test(
                self._df_clean, self._covariates, self._running_var, self._cutoff
            )
            for cov_name, cov_result in cov_balance.items():
                if not cov_result["balanced"]:
                    violations.append(
                        f"Covariate '{cov_name}' shows imbalance at cutoff "
                        f"(est = {cov_result['estimate']:.4f}, "
                        f"p = {cov_result['p_value']:.4f})"
                    )

            # For fuzzy RDD, check first stage
            if self.fuzzy and hasattr(self, '_first_stage_f'):
                if self._first_stage_f < 10:
                    violations.append(
                        f"Weak first stage in fuzzy RDD: F = {self._first_stage_f:.2f}"
                    )

        return violations
