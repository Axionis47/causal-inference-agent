"""Propensity Score Methods: PSM, IPW, AIPW."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from .base import BaseCausalMethod, MethodResult


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

        # Estimate propensity scores
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, T)
        self._propensity_scores = ps_model.predict_proba(X)[:, 1]

        # Clip propensity scores to avoid extreme weights
        self._propensity_scores = np.clip(self._propensity_scores, 0.01, 0.99)

        # Perform matching
        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]

        # Use nearest neighbor matching
        ps_treated = self._propensity_scores[treated_idx].reshape(-1, 1)
        ps_control = self._propensity_scores[control_idx].reshape(-1, 1)

        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
        nn.fit(ps_control)
        distances, indices = nn.kneighbors(ps_treated)

        # Apply caliper if specified
        if self.caliper:
            ps_std = self._propensity_scores.std()
            caliper_dist = self.caliper * ps_std
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

        self._fitted = True
        return self

    def estimate(self) -> MethodResult:
        """Compute ATT estimate.

        Returns:
            MethodResult with estimate and statistics
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        # ATT = mean(Y_treated) - mean(Y_matched_control)
        att = np.mean(self._Y_treated) - np.mean(self._Y_control)

        # Bootstrap for standard error
        n_bootstrap = 500
        bootstrap_estimates = []
        n = len(self._Y_treated)

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            boot_att = np.mean(self._Y_treated[idx]) - np.mean(self._Y_control[idx])
            bootstrap_estimates.append(boot_att)

        se = np.std(bootstrap_estimates)
        ci_lower, ci_upper = np.percentile(bootstrap_estimates, [2.5, 97.5])
        p_value = self._compute_p_value(att, se)

        # Diagnostics
        diagnostics = {
            "n_matched": self._n_treated,
            "n_unmatched": self._n_unmatched,
            "match_rate": self._n_treated / (self._n_treated + self._n_unmatched),
            "ps_mean_treated": float(np.mean(self._propensity_scores[self._matched_treated_idx])),
            "ps_mean_control": float(np.mean(self._propensity_scores[self._matched_control_idx])),
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

    def estimate(self) -> MethodResult:
        """Compute ATE estimate using IPW."""
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        ps = self._propensity_scores
        T = self._T
        Y = self._Y

        # IPW estimator
        weights_treated = T / ps
        weights_control = (1 - T) / (1 - ps)

        # Normalize weights
        weights_treated = weights_treated / weights_treated.sum() * len(T)
        weights_control = weights_control / weights_control.sum() * len(T)

        ate = np.mean(weights_treated * Y) - np.mean(weights_control * Y)

        # Bootstrap for SE
        n_bootstrap = 500
        bootstrap_estimates = []
        n = len(Y)

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            w_t = weights_treated[idx]
            w_c = weights_control[idx]
            y_b = Y[idx]
            boot_ate = np.mean(w_t * y_b) - np.mean(w_c * y_b)
            bootstrap_estimates.append(boot_ate)

        se = np.std(bootstrap_estimates)
        ci_lower, ci_upper = np.percentile(bootstrap_estimates, [2.5, 97.5])
        p_value = self._compute_p_value(ate, se)

        # Diagnostics
        diagnostics = {
            "ps_min": float(ps.min()),
            "ps_max": float(ps.max()),
            "ps_mean": float(ps.mean()),
            "effective_sample_size": float(1 / np.sum((weights_treated / weights_treated.sum())**2)),
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
        **kwargs: Any,
    ) -> "AIPWMethod":
        """Fit AIPW model."""
        if not covariates:
            raise ValueError("AIPW requires covariates")

        from sklearn.linear_model import LinearRegression

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + covariates)
        self._T = self._binarize_treatment(df_clean[treatment_col]).values
        self._Y = df_clean[outcome_col].values

        # Get covariates
        valid_covs = [c for c in covariates if c in df_clean.columns]
        self._X = df_clean[valid_covs].select_dtypes(include=[np.number]).values

        # Estimate propensity scores
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(self._X, self._T)
        self._ps = ps_model.predict_proba(self._X)[:, 1]
        self._ps = np.clip(self._ps, self.trim_threshold, 1 - self.trim_threshold)

        # Fit outcome models
        treated_idx = self._T == 1
        control_idx = self._T == 0

        self._mu1_model = LinearRegression()
        self._mu0_model = LinearRegression()

        self._mu1_model.fit(self._X[treated_idx], self._Y[treated_idx])
        self._mu0_model.fit(self._X[control_idx], self._Y[control_idx])

        # Predict potential outcomes
        self._mu1 = self._mu1_model.predict(self._X)
        self._mu0 = self._mu0_model.predict(self._X)

        self._n_treated = int(self._T.sum())
        self._n_control = int(len(self._T) - self._T.sum())

        self._fitted = True
        return self

    def estimate(self) -> MethodResult:
        """Compute ATE estimate using AIPW."""
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        T = self._T
        Y = self._Y
        ps = self._ps
        mu1 = self._mu1
        mu0 = self._mu0

        # AIPW estimator
        aipw_treated = T * (Y - mu1) / ps + mu1
        aipw_control = (1 - T) * (Y - mu0) / (1 - ps) + mu0

        ate = np.mean(aipw_treated - aipw_control)

        # Influence function based SE
        influence = aipw_treated - aipw_control - ate
        se = np.sqrt(np.var(influence) / len(influence))

        ci_lower, ci_upper = self._compute_ci(ate, se)
        p_value = self._compute_p_value(ate, se)

        diagnostics = {
            "ps_overlap": float((ps > 0.1).mean() * (ps < 0.9).mean()),
            "outcome_r2_treated": float(1 - np.var(Y[T==1] - mu1[T==1]) / np.var(Y[T==1])) if np.var(Y[T==1]) > 0 else 0,
            "outcome_r2_control": float(1 - np.var(Y[T==0] - mu0[T==0]) / np.var(Y[T==0])) if np.var(Y[T==0]) > 0 else 0,
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
            details={"trim_threshold": self.trim_threshold},
        )

        return self._result
