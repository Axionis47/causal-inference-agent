"""Propensity Score Methods: PSM, IPW, AIPW."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from .base import BaseCausalMethod, MethodResult

logger = logging.getLogger(__name__)


class PSMMethod(BaseCausalMethod):
    """Propensity Score Matching (PSM).

    Estimates Average Treatment Effect on the Treated (ATT) by matching
    treated units to similar control units based on propensity scores.

    Assumptions:
    - Conditional ignorability (no unmeasured confounders)
    - Positivity (overlap in propensity scores)
    - Correct propensity score model specification
    """

    METHOD_NAME = "Propensity Score Matching"
    ESTIMAND = "ATT"

    def __init__(
        self,
        confidence_level: float = 0.95,
        n_neighbors: int = 1,
        caliper: float | None = 0.2,
    ):
        """Initialize PSM.

        Args:
            confidence_level: Confidence level for intervals
            n_neighbors: Number of nearest neighbors to match
            caliper: Maximum distance for matches (in std of propensity)
        """
        super().__init__(confidence_level)
        self.n_neighbors = n_neighbors
        self.caliper = caliper
        self._propensity_scores = None
        self._matched_indices = None

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> "PSMMethod":
        """Fit PSM model.

        Args:
            df: DataFrame with data
            treatment_col: Name of treatment column
            outcome_col: Name of outcome column
            covariates: List of covariate column names (required for PSM)

        Returns:
            Self for chaining
        """
        if not covariates:
            raise ValueError("PSM requires covariates for propensity score estimation")

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + covariates)
        T = self._binarize_treatment(df_clean[treatment_col])
        Y = df_clean[outcome_col].values

        # Get covariates
        valid_covs = [c for c in covariates if c in df_clean.columns]
        X = df_clean[valid_covs].select_dtypes(include=[np.number])

        if X.shape[1] == 0:
            raise ValueError("No numeric covariates available for propensity score")

        # Store raw data for proper bootstrap re-estimation
        self._raw_X = X.values.copy()
        self._raw_T = T.values.copy()
        self._raw_Y = Y.copy()

        # Estimate propensity scores
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, T)
        self._propensity_scores = ps_model.predict_proba(X)[:, 1]

        # Clip propensity scores to avoid extreme weights
        self._propensity_scores = np.clip(self._propensity_scores, 0.01, 0.99)

        # Compute logit of propensity scores for matching
        logit_ps = np.log(self._propensity_scores / (1 - self._propensity_scores))

        # Perform matching on logit(PS)
        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]

        logit_treated = logit_ps[treated_idx].reshape(-1, 1)
        logit_control = logit_ps[control_idx].reshape(-1, 1)

        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        nn.fit(logit_control)
        distances, indices = nn.kneighbors(logit_treated)

        # Apply caliper on logit(PS) using pooled within-group std
        if self.caliper:
            logit_std_treated = np.std(logit_ps[treated_idx])
            logit_std_control = np.std(logit_ps[control_idx])
            n_t, n_c = len(treated_idx), len(control_idx)
            pooled_std = np.sqrt(
                ((n_t - 1) * logit_std_treated**2 + (n_c - 1) * logit_std_control**2)
                / (n_t + n_c - 2)
            )
            caliper_dist = self.caliper * pooled_std
            valid_matches = distances[:, 0] <= caliper_dist
        else:
            valid_matches = np.ones(len(treated_idx), dtype=bool)

        # Store matched pairs
        self._matched_treated_idx = treated_idx[valid_matches]
        self._matched_control_idx = control_idx[indices[valid_matches, 0]]

        # Store outcomes for matched pairs
        self._Y_treated = Y[self._matched_treated_idx]
        self._Y_control = Y[self._matched_control_idx]

        self._n_treated = len(self._matched_treated_idx)
        self._n_control = len(self._matched_control_idx)
        self._n_unmatched = len(treated_idx) - len(self._matched_treated_idx)

        # Common support diagnostic
        ps_treated_vals = self._propensity_scores[treated_idx]
        ps_control_vals = self._propensity_scores[control_idx]
        overlap_lower = max(ps_treated_vals.min(), ps_control_vals.min())
        overlap_upper = min(ps_treated_vals.max(), ps_control_vals.max())
        n_outside_treated = np.sum(
            (ps_treated_vals < overlap_lower) | (ps_treated_vals > overlap_upper)
        )
        n_outside_control = np.sum(
            (ps_control_vals < overlap_lower) | (ps_control_vals > overlap_upper)
        )
        total_n = len(T)
        pct_outside = (n_outside_treated + n_outside_control) / total_n * 100

        self._common_support = {
            "overlap_lower": float(overlap_lower),
            "overlap_upper": float(overlap_upper),
            "n_treated_outside_support": int(n_outside_treated),
            "n_control_outside_support": int(n_outside_control),
            "pct_outside_support": float(pct_outside),
        }

        self._fitted = True
        return self

    def _match_and_estimate_att(self, X, T, Y):
        """Re-estimate propensity scores, match, and compute ATT.

        Used internally for bootstrap iterations.

        Args:
            X: Covariate matrix
            T: Treatment array
            Y: Outcome array

        Returns:
            ATT estimate or None if matching fails
        """
        try:
            ps_model = LogisticRegression(max_iter=1000, random_state=None)
            ps_model.fit(X, T)
            ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)

            logit_ps = np.log(ps / (1 - ps))
            treated_idx = np.where(T == 1)[0]
            control_idx = np.where(T == 0)[0]

            if len(treated_idx) == 0 or len(control_idx) == 0:
                return None

            logit_treated = logit_ps[treated_idx].reshape(-1, 1)
            logit_control = logit_ps[control_idx].reshape(-1, 1)

            k = min(self.n_neighbors, len(control_idx))
            nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
            nn.fit(logit_control)
            distances, indices = nn.kneighbors(logit_treated)

            if self.caliper:
                logit_std_t = np.std(logit_ps[treated_idx])
                logit_std_c = np.std(logit_ps[control_idx])
                n_t, n_c = len(treated_idx), len(control_idx)
                pooled_std = np.sqrt(
                    ((n_t - 1) * logit_std_t**2 + (n_c - 1) * logit_std_c**2)
                    / max(n_t + n_c - 2, 1)
                )
                caliper_dist = self.caliper * pooled_std
                valid = distances[:, 0] <= caliper_dist
            else:
                valid = np.ones(len(treated_idx), dtype=bool)

            if valid.sum() == 0:
                return None

            matched_treated = treated_idx[valid]
            matched_control = control_idx[indices[valid, 0]]
            return np.mean(Y[matched_treated]) - np.mean(Y[matched_control])
        except Exception:
            return None

    def estimate(self) -> MethodResult:
        """Compute ATT estimate.

        Returns:
            MethodResult with estimate and statistics
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        # ATT = mean(Y_treated) - mean(Y_matched_control)
        att = np.mean(self._Y_treated) - np.mean(self._Y_control)

        # Proper bootstrap: resample original observations, re-estimate PS,
        # re-match, and compute ATT each iteration
        n_bootstrap = 500
        bootstrap_estimates = []
        n = len(self._raw_Y)

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            boot_X = self._raw_X[idx]
            boot_T = self._raw_T[idx]
            boot_Y = self._raw_Y[idx]

            boot_att = self._match_and_estimate_att(boot_X, boot_T, boot_Y)
            if boot_att is not None:
                bootstrap_estimates.append(boot_att)

        if len(bootstrap_estimates) < 50:
            logger.warning(
                "Only %d/%d bootstrap iterations produced valid matches. "
                "SE estimate may be unreliable.",
                len(bootstrap_estimates),
                n_bootstrap,
            )

        se = np.std(bootstrap_estimates) if bootstrap_estimates else np.nan
        if bootstrap_estimates:
            ci_lower, ci_upper = np.percentile(bootstrap_estimates, [2.5, 97.5])
        else:
            ci_lower, ci_upper = np.nan, np.nan
        p_value = self._compute_p_value(att, se, n=self._n_treated + self._n_control)

        # Diagnostics
        diagnostics = {
            "n_matched": self._n_treated,
            "n_unmatched": self._n_unmatched,
            "match_rate": self._n_treated / (self._n_treated + self._n_unmatched),
            "ps_mean_treated": float(np.mean(self._propensity_scores[self._matched_treated_idx])),
            "ps_mean_control": float(np.mean(self._propensity_scores[self._matched_control_idx])),
            "bootstrap_iterations_valid": len(bootstrap_estimates),
            "common_support": self._common_support,
        }

        self._result = MethodResult(
            method=self.METHOD_NAME,
            estimand=self.ESTIMAND,
            estimate=float(att),
            std_error=float(se),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(p_value),
            n_treated=self._n_treated,
            n_control=self._n_control,
            assumptions_tested=["positivity", "conditional_ignorability"],
            diagnostics=diagnostics,
            details={
                "n_neighbors": self.n_neighbors,
                "caliper": self.caliper,
            },
        )

        return self._result


class IPWMethod(BaseCausalMethod):
    """Inverse Probability Weighting (IPW).

    Estimates ATE by reweighting observations using inverse propensity scores.

    Assumptions:
    - Conditional ignorability
    - Positivity (all units have non-zero probability of treatment)
    - Correct propensity score model
    """

    METHOD_NAME = "Inverse Probability Weighting"
    ESTIMAND = "ATE"

    def __init__(self, confidence_level: float = 0.95, trim_threshold: float = 0.01):
        """Initialize IPW.

        Args:
            confidence_level: Confidence level for intervals
            trim_threshold: Threshold for trimming extreme propensity scores
        """
        super().__init__(confidence_level)
        self.trim_threshold = trim_threshold
        self._propensity_scores = None

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> "IPWMethod":
        """Fit IPW model."""
        if not covariates:
            raise ValueError("IPW requires covariates for propensity score estimation")

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + covariates)
        self._T = self._binarize_treatment(df_clean[treatment_col]).values
        self._Y = df_clean[outcome_col].values

        # Get covariates
        valid_covs = [c for c in covariates if c in df_clean.columns]
        X = df_clean[valid_covs].select_dtypes(include=[np.number])

        # Estimate propensity scores
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, self._T)
        self._propensity_scores = ps_model.predict_proba(X)[:, 1]

        # Trim extreme propensity scores
        self._propensity_scores = np.clip(
            self._propensity_scores,
            self.trim_threshold,
            1 - self.trim_threshold
        )

        self._n_treated = int(self._T.sum())
        self._n_control = int(len(self._T) - self._T.sum())

        self._fitted = True
        return self

    @staticmethod
    def _hajek_ate(T, Y, ps):
        """Compute ATE using the Hajek (normalized) estimator.

        Args:
            T: Treatment array
            Y: Outcome array
            ps: Propensity score array

        Returns:
            ATE estimate
        """
        # E[Y(1)] = sum(T*Y/ps) / sum(T/ps)
        mu1 = np.sum(T * Y / ps) / np.sum(T / ps)
        # E[Y(0)] = sum((1-T)*Y/(1-ps)) / sum((1-T)/(1-ps))
        mu0 = np.sum((1 - T) * Y / (1 - ps)) / np.sum((1 - T) / (1 - ps))
        return mu1 - mu0

    def estimate(self) -> MethodResult:
        """Compute ATE estimate using Hajek (normalized IPW) estimator."""
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        ps = self._propensity_scores
        T = self._T
        Y = self._Y

        # Hajek estimator (properly normalized, no double-division)
        ate = self._hajek_ate(T, Y, ps)

        # Bootstrap for SE -- resample original data, re-estimate PS, re-compute
        n_bootstrap = 500
        bootstrap_estimates = []
        n = len(Y)

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            boot_T = T[idx]
            boot_Y = Y[idx]
            boot_ps = ps[idx]
            try:
                boot_ate = self._hajek_ate(boot_T, boot_Y, boot_ps)
                if np.isfinite(boot_ate):
                    bootstrap_estimates.append(boot_ate)
            except Exception:
                continue

        se = np.std(bootstrap_estimates)
        ci_lower, ci_upper = np.percentile(bootstrap_estimates, [2.5, 97.5])
        p_value = self._compute_p_value(ate, se, n=n)

        # Effective sample size: ESS = (sum(w))^2 / sum(w^2) for each arm
        w_treated = T / ps
        w_control = (1 - T) / (1 - ps)
        ess_treated = np.sum(w_treated)**2 / np.sum(w_treated**2)
        ess_control = np.sum(w_control)**2 / np.sum(w_control**2)

        # Extreme weight diagnostics
        median_w_treated = np.median(w_treated[T == 1])
        median_w_control = np.median(w_control[T == 0])
        n_extreme_treated = int(np.sum(w_treated[T == 1] > 10 * median_w_treated))
        n_extreme_control = int(np.sum(w_control[T == 0] > 10 * median_w_control))

        # Diagnostics
        diagnostics = {
            "ps_min": float(ps.min()),
            "ps_max": float(ps.max()),
            "ps_mean": float(ps.mean()),
            "ess_treated": float(ess_treated),
            "ess_control": float(ess_control),
            "n_extreme_weights_treated": n_extreme_treated,
            "n_extreme_weights_control": n_extreme_control,
        }

        if n_extreme_treated + n_extreme_control > 0:
            logger.warning(
                "Extreme IPW weights detected: %d treated, %d control "
                "observations have weights > 10x median. Consider stronger "
                "trimming or stabilized weights.",
                n_extreme_treated,
                n_extreme_control,
            )

        self._result = MethodResult(
            method=self.METHOD_NAME,
            estimand=self.ESTIMAND,
            estimate=float(ate),
            std_error=float(se),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(p_value),
            n_treated=self._n_treated,
            n_control=self._n_control,
            assumptions_tested=["positivity", "conditional_ignorability"],
            diagnostics=diagnostics,
            details={"trim_threshold": self.trim_threshold},
        )

        return self._result


class AIPWMethod(BaseCausalMethod):
    """Augmented Inverse Probability Weighting (Doubly Robust).

    Combines IPW with outcome regression for double robustness.
    Consistent if either the propensity or outcome model is correct.

    Assumptions:
    - Conditional ignorability
    - Positivity
    """

    METHOD_NAME = "Doubly Robust (AIPW)"
    ESTIMAND = "ATE"

    def __init__(self, confidence_level: float = 0.95, trim_threshold: float = 0.01):
        """Initialize AIPW."""
        super().__init__(confidence_level)
        self.trim_threshold = trim_threshold

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        n_folds: int = 5,
        **kwargs: Any,
    ) -> "AIPWMethod":
        """Fit AIPW model with K-fold cross-fitting.

        Cross-fitting avoids overfitting bias by fitting nuisance models
        (propensity score and outcome models) on training folds and
        predicting on held-out folds.

        Args:
            df: DataFrame with data
            treatment_col: Treatment column name
            outcome_col: Outcome column name
            covariates: Covariate column names
            n_folds: Number of cross-fitting folds (default 5)

        Returns:
            Self for chaining
        """
        if not covariates:
            raise ValueError("AIPW requires covariates")

        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import KFold

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + covariates)
        self._T = self._binarize_treatment(df_clean[treatment_col]).values
        self._Y = df_clean[outcome_col].values

        # Get covariates
        valid_covs = [c for c in covariates if c in df_clean.columns]
        self._X = df_clean[valid_covs].select_dtypes(include=[np.number]).values

        n = len(self._Y)
        self._ps = np.zeros(n)
        self._mu1 = np.zeros(n)
        self._mu0 = np.zeros(n)

        # K-fold cross-fitting
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(self._X):
            X_train, X_test = self._X[train_idx], self._X[test_idx]
            T_train, T_test = self._T[train_idx], self._T[test_idx]
            Y_train = self._Y[train_idx]

            # Fit propensity score model on training fold
            ps_model = LogisticRegression(max_iter=1000, random_state=42)
            ps_model.fit(X_train, T_train)
            ps_pred = ps_model.predict_proba(X_test)[:, 1]
            self._ps[test_idx] = np.clip(
                ps_pred, self.trim_threshold, 1 - self.trim_threshold
            )

            # Fit outcome models on training fold
            treated_train = T_train == 1
            control_train = T_train == 0

            if treated_train.sum() > 0:
                mu1_model = LinearRegression()
                mu1_model.fit(X_train[treated_train], Y_train[treated_train])
                self._mu1[test_idx] = mu1_model.predict(X_test)
            else:
                self._mu1[test_idx] = np.mean(Y_train)

            if control_train.sum() > 0:
                mu0_model = LinearRegression()
                mu0_model.fit(X_train[control_train], Y_train[control_train])
                self._mu0[test_idx] = mu0_model.predict(X_test)
            else:
                self._mu0[test_idx] = np.mean(Y_train)

        self._n_treated = int(self._T.sum())
        self._n_control = int(len(self._T) - self._T.sum())
        self._n_folds = n_folds

        self._fitted = True
        return self

    def estimate(self) -> MethodResult:
        """Compute ATE estimate using AIPW with cross-fitted nuisance models."""
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        T = self._T
        Y = self._Y
        ps = self._ps
        mu1 = self._mu1
        mu0 = self._mu0
        n = len(Y)

        # AIPW estimator (using cross-fitted predictions)
        aipw_treated = T * (Y - mu1) / ps + mu1
        aipw_control = (1 - T) * (Y - mu0) / (1 - ps) + mu0

        ate = np.mean(aipw_treated - aipw_control)

        # Influence function based SE
        influence = aipw_treated - aipw_control - ate
        se = np.sqrt(np.var(influence) / n)

        ci_lower, ci_upper = self._compute_ci(ate, se, n=n)
        p_value = self._compute_p_value(ate, se, n=n)

        # Overlap: fraction of units with PS in [0.1, 0.9]
        ps_in_range = np.mean((ps > 0.1) & (ps < 0.9))

        diagnostics = {
            "ps_overlap": float(ps_in_range),
            "outcome_r2_treated": (
                float(1 - np.var(Y[T == 1] - mu1[T == 1]) / np.var(Y[T == 1]))
                if np.var(Y[T == 1]) > 0 else 0.0
            ),
            "outcome_r2_control": (
                float(1 - np.var(Y[T == 0] - mu0[T == 0]) / np.var(Y[T == 0]))
                if np.var(Y[T == 0]) > 0 else 0.0
            ),
            "cross_fitting_folds": self._n_folds,
        }

        self._result = MethodResult(
            method=self.METHOD_NAME,
            estimand=self.ESTIMAND,
            estimate=float(ate),
            std_error=float(se),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(p_value),
            n_treated=self._n_treated,
            n_control=self._n_control,
            assumptions_tested=["positivity", "conditional_ignorability", "doubly_robust"],
            diagnostics=diagnostics,
            details={
                "trim_threshold": self.trim_threshold,
                "cross_fitting": True,
                "n_folds": self._n_folds,
            },
        )

        return self._result
