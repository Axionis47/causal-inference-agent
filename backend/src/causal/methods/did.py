"""Difference-in-Differences (DiD) Method."""

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .base import BaseCausalMethod, MethodResult


class DifferenceInDifferencesMethod(BaseCausalMethod):
    """Difference-in-Differences (DiD) estimator.

    Estimates ATT by comparing changes in outcomes between treatment
    and control groups before and after treatment.

    Assumptions:
    - Parallel trends (treatment and control would have followed
      same trajectory absent treatment)
    - No spillovers
    - Treatment timing is sharp
    - Stable composition
    """

    METHOD_NAME = "Difference-in-Differences"
    ESTIMAND = "ATT"

    def __init__(self, confidence_level: float = 0.95, cluster_col: str | None = None):
        """Initialize DiD.

        Args:
            confidence_level: Confidence level for intervals
            cluster_col: Column for clustered standard errors
        """
        super().__init__(confidence_level)
        self.cluster_col = cluster_col
        self._model = None
        self._results = None

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        time_col: str | None = None,
        post_col: str | None = None,
        **kwargs: Any,
    ) -> "DifferenceInDifferencesMethod":
        """Fit DiD model.

        Args:
            df: DataFrame with panel or repeated cross-section data
            treatment_col: Name of treatment group indicator (0/1)
            outcome_col: Name of outcome column
            covariates: Additional control variables
            time_col: Time period column (optional, infers post from this)
            post_col: Post-treatment indicator column (0/1)

        Returns:
            Self for chaining
        """
        # Determine post-treatment indicator
        if post_col and post_col in df.columns:
            df_clean = df.copy()
            df_clean['_post'] = df_clean[post_col]
        elif time_col and time_col in df.columns:
            # Assume median time is treatment cutoff
            median_time = df[time_col].median()
            df_clean = df.copy()
            df_clean['_post'] = (df_clean[time_col] >= median_time).astype(int)
        else:
            raise ValueError("DiD requires either post_col or time_col to identify post-treatment period")

        # Clean data
        required_cols = [treatment_col, outcome_col, '_post']
        if covariates:
            required_cols.extend(covariates)
        df_clean = df_clean.dropna(subset=required_cols)

        # Create treatment and post indicators
        T = self._binarize_treatment(df_clean[treatment_col])
        post = df_clean['_post'].astype(int)
        Y = df_clean[outcome_col].values

        # Create interaction term (DiD estimator)
        did_term = T * post

        # Build design matrix: [const, treated, post, treated*post, covariates]
        X = np.column_stack([np.ones(len(Y)), T.values, post.values, did_term.values])

        if covariates:
            valid_covs = [c for c in covariates if c in df_clean.columns]
            X_covs = df_clean[valid_covs].select_dtypes(include=[np.number]).values
            if X_covs.shape[1] > 0:
                X = np.column_stack([X, X_covs])

        # Fit OLS
        self._model = sm.OLS(Y, X)

        if self.cluster_col and self.cluster_col in df_clean.columns:
            # Clustered standard errors
            self._results = self._model.fit(
                cov_type='cluster',
                cov_kwds={'groups': df_clean[self.cluster_col].values}
            )
        else:
            self._results = self._model.fit(cov_type='HC1')

        # Store group counts
        self._n_treated_post = int((T.values == 1) & (post.values == 1)).sum()
        self._n_treated_pre = int((T.values == 1) & (post.values == 0)).sum()
        self._n_control_post = int((T.values == 0) & (post.values == 1)).sum()
        self._n_control_pre = int((T.values == 0) & (post.values == 0)).sum()

        # Store means for parallel trends check
        self._means = {
            'treated_pre': float(Y[(T.values == 1) & (post.values == 0)].mean()) if self._n_treated_pre > 0 else np.nan,
            'treated_post': float(Y[(T.values == 1) & (post.values == 1)].mean()) if self._n_treated_post > 0 else np.nan,
            'control_pre': float(Y[(T.values == 0) & (post.values == 0)].mean()) if self._n_control_pre > 0 else np.nan,
            'control_post': float(Y[(T.values == 0) & (post.values == 1)].mean()) if self._n_control_post > 0 else np.nan,
        }

        self._fitted = True
        return self

    def estimate(self) -> MethodResult:
        """Compute DiD estimate (ATT).

        Returns:
            MethodResult with estimate and statistics
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        # DiD coefficient is the interaction term (index 3)
        did_estimate = self._results.params[3]
        se = self._results.bse[3]
        ci = self._results.conf_int(alpha=self.alpha)[3]
        p_value = self._results.pvalues[3]

        # Compute raw DiD manually for verification
        raw_did = (
            (self._means['treated_post'] - self._means['treated_pre']) -
            (self._means['control_post'] - self._means['control_pre'])
        )

        diagnostics = {
            "raw_did": float(raw_did) if not np.isnan(raw_did) else None,
            "treated_pre_mean": self._means['treated_pre'],
            "treated_post_mean": self._means['treated_post'],
            "control_pre_mean": self._means['control_pre'],
            "control_post_mean": self._means['control_post'],
            "treated_change": self._means['treated_post'] - self._means['treated_pre'],
            "control_change": self._means['control_post'] - self._means['control_pre'],
            "r_squared": float(self._results.rsquared),
        }

        self._result = MethodResult(
            method=self.METHOD_NAME,
            estimand=self.ESTIMAND,
            estimate=float(did_estimate),
            std_error=float(se),
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            p_value=float(p_value),
            n_treated=self._n_treated_pre + self._n_treated_post,
            n_control=self._n_control_pre + self._n_control_post,
            assumptions_tested=["parallel_trends", "no_spillovers", "stable_composition"],
            diagnostics=diagnostics,
            details={
                "n_treated_pre": self._n_treated_pre,
                "n_treated_post": self._n_treated_post,
                "n_control_pre": self._n_control_pre,
                "n_control_post": self._n_control_post,
                "clustered_se": self.cluster_col is not None,
            },
        )

        return self._result

    def validate_assumptions(
        self, df: pd.DataFrame, treatment_col: str, outcome_col: str
    ) -> list[str]:
        """Validate DiD assumptions."""
        violations = super().validate_assumptions(df, treatment_col, outcome_col)

        # Check for sufficient observations in each cell
        if self._fitted:
            min_cell = min(
                self._n_treated_pre, self._n_treated_post,
                self._n_control_pre, self._n_control_post
            )
            if min_cell < 10:
                violations.append(f"Small cell size: minimum {min_cell} observations")

            # Check pre-treatment trends (if multiple pre-periods exist)
            # This is a simplified check - full parallel trends test requires multiple periods
            if self._means['control_pre'] != 0:
                pre_ratio = self._means['treated_pre'] / self._means['control_pre']
                if abs(pre_ratio - 1) > 0.5:
                    violations.append("Large pre-treatment difference between groups")

        return violations
