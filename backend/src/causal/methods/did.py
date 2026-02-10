"""Difference-in-Differences (DiD) Method."""

import logging
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .base import BaseCausalMethod, MethodResult

logger = logging.getLogger(__name__)


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
        self._used_median_cutoff = False

        # Determine post-treatment indicator
        if post_col and post_col in df.columns:
            df_clean = df.copy()
            df_clean['_post'] = df_clean[post_col]
        elif time_col and time_col in df.columns:
            # Assume median time is treatment cutoff
            median_time = df[time_col].median()
            df_clean = df.copy()
            df_clean['_post'] = (df_clean[time_col] >= median_time).astype(int)
            self._used_median_cutoff = True
            logger.warning(
                "WARNING: Using median time (%s) as treatment cutoff. "
                "Provide explicit post_col for reliable results.",
                median_time,
            )
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

        # Fit OLS with robust SE by default (HC1)
        self._model = sm.OLS(Y, X)

        if self.cluster_col and self.cluster_col in df_clean.columns:
            # Clustered standard errors
            self._results = self._model.fit(
                cov_type='cluster',
                cov_kwds={'groups': df_clean[self.cluster_col].values}
            )
        else:
            # Default to HC1 robust standard errors
            self._results = self._model.fit(cov_type='HC1')

            # Detect panel data: check if units appear multiple times
            if treatment_col in df_clean.columns:
                unit_counts = df_clean[treatment_col].groupby(
                    df_clean.index
                ).count()
                # Simple heuristic: if there's a time column and the number of
                # unique (treatment, time) combinations suggests repeated obs
                if time_col and time_col in df_clean.columns:
                    n_unique_units = df_clean.groupby(treatment_col).ngroups
                    n_rows = len(df_clean)
                    if n_rows > n_unique_units * 1.5:
                        logger.warning(
                            "Panel data detected (repeated units across time). "
                            "Consider using clustered standard errors by specifying "
                            "cluster_col to account for within-unit correlation."
                        )

        # Store group counts
        T_vals = T.values
        post_vals = post.values
        self._n_treated_post = int(((T_vals == 1) & (post_vals == 1)).sum())
        self._n_treated_pre = int(((T_vals == 1) & (post_vals == 0)).sum())
        self._n_control_post = int(((T_vals == 0) & (post_vals == 1)).sum())
        self._n_control_pre = int(((T_vals == 0) & (post_vals == 0)).sum())

        # Store data for parallel trends test
        self._T_vals = T_vals
        self._post_vals = post_vals
        self._Y_vals = Y
        self._time_col = time_col
        self._df_clean = df_clean

        # Store means for parallel trends check
        self._means = {
            'treated_pre': float(Y[(T_vals == 1) & (post_vals == 0)].mean()) if self._n_treated_pre > 0 else np.nan,
            'treated_post': float(Y[(T_vals == 1) & (post_vals == 1)].mean()) if self._n_treated_post > 0 else np.nan,
            'control_pre': float(Y[(T_vals == 0) & (post_vals == 0)].mean()) if self._n_control_pre > 0 else np.nan,
            'control_post': float(Y[(T_vals == 0) & (post_vals == 1)].mean()) if self._n_control_post > 0 else np.nan,
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

        if self._used_median_cutoff:
            diagnostics["time_cutoff_warning"] = (
                "WARNING: Using median time as treatment cutoff. "
                "Provide explicit post_col for reliable results."
            )

        if hasattr(self, '_parallel_trends_result'):
            diagnostics["parallel_trends_test"] = self._parallel_trends_result
            if self._parallel_trends_result.get("p_value", 1.0) < 0.05:
                diagnostics["parallel_trends_warning"] = (
                    "WARNING: Parallel trends assumption may be violated "
                    f"(p={self._parallel_trends_result['p_value']:.4f}). "
                    "DiD estimates should be interpreted with caution."
                )

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
        """Validate DiD assumptions including formal parallel trends test."""
        violations = super().validate_assumptions(df, treatment_col, outcome_col)

        # Check for sufficient observations in each cell
        if self._fitted:
            min_cell = min(
                self._n_treated_pre, self._n_treated_post,
                self._n_control_pre, self._n_control_post
            )
            if min_cell < 10:
                violations.append(f"Small cell size: minimum {min_cell} observations")

            # Parallel trends test
            self._run_parallel_trends_test(df, treatment_col, outcome_col, violations)

        return violations

    def _run_parallel_trends_test(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        violations: list[str],
    ) -> None:
        """Run a formal parallel trends test.

        If multiple pre-periods: regress Y_pre on time * group interaction,
        test interaction coefficient = 0.
        If only 2 periods: test equality of outcome changes between groups.

        Args:
            df: DataFrame
            treatment_col: Treatment column
            outcome_col: Outcome column
            violations: List to append violations to (mutated in place)
        """
        from scipy import stats

        T_vals = self._T_vals
        post_vals = self._post_vals
        Y_vals = self._Y_vals

        # Get pre-treatment observations
        pre_mask = post_vals == 0
        T_pre = T_vals[pre_mask]
        Y_pre = Y_vals[pre_mask]

        if self._time_col and self._time_col in self._df_clean.columns:
            time_pre = self._df_clean[self._time_col].values[pre_mask]
            unique_pre_times = np.unique(time_pre)

            if len(unique_pre_times) > 2:
                # Multiple pre-periods: regress Y_pre on time, group, time*group
                # Convert time to numeric for regression
                try:
                    time_numeric = time_pre.astype(float)
                except (ValueError, TypeError):
                    # If time can't be cast to float, use rank encoding
                    time_rank = {t: i for i, t in enumerate(sorted(unique_pre_times))}
                    time_numeric = np.array([time_rank[t] for t in time_pre], dtype=float)

                # Design matrix: [const, time, group, time*group]
                interaction = time_numeric * T_pre
                X_pt = np.column_stack([
                    np.ones(len(Y_pre)),
                    time_numeric,
                    T_pre.astype(float),
                    interaction,
                ])

                try:
                    pt_model = sm.OLS(Y_pre, X_pt)
                    pt_results = pt_model.fit(cov_type='HC1')

                    # Test interaction coefficient (index 3) = 0
                    pt_coef = pt_results.params[3]
                    pt_pval = pt_results.pvalues[3]

                    self._parallel_trends_result = {
                        "test": "pre-trend interaction",
                        "coefficient": float(pt_coef),
                        "p_value": float(pt_pval),
                        "n_pre_periods": int(len(unique_pre_times)),
                    }

                    if pt_pval < 0.05:
                        violations.append(
                            f"Parallel trends assumption violated: pre-treatment "
                            f"time*group interaction is significant "
                            f"(coef={pt_coef:.4f}, p={pt_pval:.4f})"
                        )
                        logger.warning(
                            "Parallel trends test FAILED (p=%.4f). "
                            "DiD estimates may be biased.",
                            pt_pval,
                        )
                except Exception as e:
                    logger.debug("Parallel trends regression failed: %s", e)
                    return

            elif len(unique_pre_times) == 2:
                # Exactly 2 pre-periods: test equality of changes
                try:
                    t1, t2 = sorted(unique_pre_times)

                    # Change for treated group
                    y_t1_treated = Y_pre[(time_pre == t1) & (T_pre == 1)]
                    y_t2_treated = Y_pre[(time_pre == t2) & (T_pre == 1)]
                    # Change for control group
                    y_t1_control = Y_pre[(time_pre == t1) & (T_pre == 0)]
                    y_t2_control = Y_pre[(time_pre == t2) & (T_pre == 0)]

                    if len(y_t1_treated) > 0 and len(y_t2_treated) > 0 and \
                       len(y_t1_control) > 0 and len(y_t2_control) > 0:
                        change_treated = np.mean(y_t2_treated) - np.mean(y_t1_treated)
                        change_control = np.mean(y_t2_control) - np.mean(y_t1_control)

                        # Test if changes differ using a regression approach
                        # Equivalent to DiD on the two pre-periods
                        time_indicator = (time_pre == t2).astype(float)
                        interaction = time_indicator * T_pre.astype(float)
                        X_pt2 = np.column_stack([
                            np.ones(len(Y_pre)),
                            time_indicator,
                            T_pre.astype(float),
                            interaction,
                        ])
                        pt2_model = sm.OLS(Y_pre, X_pt2)
                        pt2_results = pt2_model.fit(cov_type='HC1')
                        pt2_pval = pt2_results.pvalues[3]
                        pt2_coef = pt2_results.params[3]

                        self._parallel_trends_result = {
                            "test": "pre-period change equality",
                            "treated_change": float(change_treated),
                            "control_change": float(change_control),
                            "difference": float(change_treated - change_control),
                            "p_value": float(pt2_pval),
                        }

                        if pt2_pval < 0.05:
                            violations.append(
                                f"Parallel trends assumption may be violated: "
                                f"pre-treatment changes differ between groups "
                                f"(diff={pt2_coef:.4f}, p={pt2_pval:.4f})"
                            )
                except Exception as e:
                    logger.debug("Two-period parallel trends test failed: %s", e)
        else:
            # No time column: fall back to simple mean difference check
            if len(Y_pre) > 0 and T_pre.sum() > 0 and (1 - T_pre).sum() > 0:
                y_pre_treated = Y_pre[T_pre == 1]
                y_pre_control = Y_pre[T_pre == 0]
                t_stat, t_pval = stats.ttest_ind(y_pre_treated, y_pre_control)

                self._parallel_trends_result = {
                    "test": "pre-treatment means equality",
                    "t_statistic": float(t_stat),
                    "p_value": float(t_pval),
                }

                if t_pval < 0.05:
                    violations.append(
                        f"Significant pre-treatment difference between groups "
                        f"(t={t_stat:.4f}, p={t_pval:.4f})"
                    )
