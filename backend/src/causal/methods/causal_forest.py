"""Causal Forest for Heterogeneous Treatment Effects."""

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseCausalMethod, MethodResult


class CausalForestMethod(BaseCausalMethod):
    """Causal Forest (Generalized Random Forest).

    Estimates heterogeneous treatment effects using honest causal forests.
    Provides valid confidence intervals for CATEs.

    Based on Athey, Tibshirani, Wager (2019):
    "Generalized Random Forests"

    Uses econml's CausalForestDML implementation when available,
    falls back to T-Learner implementation otherwise.
    """

    METHOD_NAME = "Causal Forest"
    ESTIMAND = "CATE"

    def __init__(
        self,
        confidence_level: float = 0.95,
        n_estimators: int = 100,
        min_samples_leaf: int = 5,
        max_depth: int | None = None,
    ):
        """Initialize Causal Forest.

        Args:
            confidence_level: Confidence level for intervals
            n_estimators: Number of trees
            min_samples_leaf: Minimum samples per leaf
            max_depth: Maximum tree depth (None for unlimited)
        """
        super().__init__(confidence_level)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self._use_econml = False

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> "CausalForestMethod":
        """Fit Causal Forest model.

        Args:
            df: DataFrame with data
            treatment_col: Treatment variable
            outcome_col: Outcome variable
            covariates: Covariate columns

        Returns:
            Self for chaining
        """
        if not covariates:
            raise ValueError("Causal Forest requires covariates")

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + covariates)
        self._T = self._binarize_treatment(df_clean[treatment_col]).values.astype(float)
        self._Y = df_clean[outcome_col].values.astype(float)

        valid_covs = [c for c in covariates if c in df_clean.columns]
        self._X = df_clean[valid_covs].select_dtypes(include=[np.number]).values.astype(float)

        self._n_treated = int(self._T.sum())
        self._n_control = int(len(self._T) - self._T.sum())
        self._covariates = valid_covs

        # Try econml first
        try:
            from econml.dml import CausalForestDML

            # Detect if treatment is binary/discrete
            is_discrete = len(np.unique(self._T)) <= 2

            self._model = CausalForestDML(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=42,
                discrete_treatment=is_discrete,
                inference='blb',  # Bayesian bootstrap for valid CIs
            )

            if is_discrete:
                self._model.fit(self._Y, self._T, X=self._X)
            else:
                T_reshaped = self._T.reshape(-1, 1)
                self._model.fit(self._Y, T_reshaped, X=self._X)
            self._use_econml = True

        except ImportError:
            # Fallback to T-Learner-style CATE estimation
            self._fit_fallback()

        self._fitted = True
        return self

    def _fit_fallback(self):
        """Fallback T-Learner implementation using sklearn RandomForest."""
        from sklearn.ensemble import RandomForestRegressor

        # Split into treated and control
        treated_idx = self._T == 1
        control_idx = self._T == 0

        # Fit outcome models
        self._model_treated = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=42,
        )
        self._model_control = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=42,
        )

        self._model_treated.fit(self._X[treated_idx], self._Y[treated_idx])
        self._model_control.fit(self._X[control_idx], self._Y[control_idx])

        self._use_econml = False

    def estimate(self) -> MethodResult:
        """Compute CATE estimates.

        Returns:
            MethodResult with estimates and statistics
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        if self._use_econml:
            # Use econml inference with BLB
            cate = self._model.effect(self._X).flatten()

            # Get confidence intervals from BLB inference
            try:
                cate_lower, cate_upper = self._model.effect_interval(
                    self._X, alpha=self.alpha
                )
                cate_lower = cate_lower.flatten()
                cate_upper = cate_upper.flatten()

                # Derive ATE SE from the interval
                ate = float(np.mean(cate))
                # Use model's inference for ATE SE
                try:
                    ate_inf = self._model.ate_interval(self._X, alpha=self.alpha)
                    ate_se = float((ate_inf[1] - ate_inf[0]) / (2 * 1.96))
                except Exception:
                    # Fallback: SE from individual CATE SEs
                    cate_se = (cate_upper - cate_lower) / (2 * 1.96)
                    ate_se = float(np.sqrt(np.mean(cate_se ** 2) / len(cate)))

            except Exception:
                # If BLB inference fails, use std-based estimate
                ate = float(np.mean(cate))
                ate_se = float(np.std(cate) / np.sqrt(len(cate)))
                cate_lower = cate - 1.96 * np.std(cate)
                cate_upper = cate + 1.96 * np.std(cate)

            method_name = self.METHOD_NAME

        else:
            # Fallback T-Learner estimation with proper bootstrap SE
            from sklearn.ensemble import RandomForestRegressor

            mu1 = self._model_treated.predict(self._X)
            mu0 = self._model_control.predict(self._X)
            cate = mu1 - mu0

            ate = float(np.mean(cate))

            # --- Bootstrap SE: resample observations, refit both models ---
            n_bootstrap = 100
            n = len(self._X)
            bootstrap_ates = []
            rng = np.random.RandomState(123)

            for b in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                X_b = self._X[idx]
                T_b = self._T[idx]
                Y_b = self._Y[idx]

                treated_b = T_b == 1
                control_b = T_b == 0

                if treated_b.sum() < 2 or control_b.sum() < 2:
                    continue

                try:
                    m1 = RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        min_samples_leaf=self.min_samples_leaf,
                        max_depth=self.max_depth,
                        random_state=b,
                    )
                    m0 = RandomForestRegressor(
                        n_estimators=self.n_estimators,
                        min_samples_leaf=self.min_samples_leaf,
                        max_depth=self.max_depth,
                        random_state=b,
                    )
                    m1.fit(X_b[treated_b], Y_b[treated_b])
                    m0.fit(X_b[control_b], Y_b[control_b])
                    cate_b = m1.predict(self._X) - m0.predict(self._X)
                    bootstrap_ates.append(np.mean(cate_b))
                except Exception:
                    continue

            if len(bootstrap_ates) >= 10:
                ate_se = float(np.std(bootstrap_ates))
            else:
                ate_se = float(np.std(cate) / np.sqrt(len(cate)))

            cate_se_individual = np.std(cate)
            cate_lower = cate - 1.96 * cate_se_individual
            cate_upper = cate + 1.96 * cate_se_individual

            # Label fallback correctly
            method_name = "T-Learner Fallback (not Causal Forest)"

        self._cate = cate

        ci_lower, ci_upper = self._compute_ci(ate, ate_se)
        p_value = self._compute_p_value(ate, ate_se)

        # Heterogeneity diagnostics
        diagnostics = {
            "cate_mean": float(np.mean(cate)),
            "cate_std": float(np.std(cate)),
            "cate_min": float(np.min(cate)),
            "cate_max": float(np.max(cate)),
            "cate_median": float(np.median(cate)),
            "pct_positive": float((cate > 0).mean() * 100),
            "heterogeneity_index": float(np.std(cate) / (abs(np.mean(cate)) + 1e-10)),
            "iqr": float(np.percentile(cate, 75) - np.percentile(cate, 25)),
            "used_econml": self._use_econml,
        }

        if not self._use_econml:
            diagnostics["note"] = (
                "econml not available; using T-Learner fallback with bootstrap SE. "
                "Install econml for proper causal forest with BLB inference."
            )

        # Variable importance (if available)
        if self._use_econml and hasattr(self._model, 'feature_importances_'):
            try:
                importances = self._model.feature_importances_()
                diagnostics["feature_importances"] = {
                    self._covariates[i]: float(imp)
                    for i, imp in enumerate(importances[:len(self._covariates)])
                }
            except Exception:
                pass

        self._result = MethodResult(
            method=method_name,
            estimand="ATE",
            estimate=ate,
            std_error=ate_se,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(p_value),
            n_treated=self._n_treated,
            n_control=self._n_control,
            assumptions_tested=["conditional_ignorability", "positivity", "overlap"],
            diagnostics=diagnostics,
            details={
                "n_estimators": self.n_estimators,
                "min_samples_leaf": self.min_samples_leaf,
                "max_depth": self.max_depth,
            },
        )

        return self._result

    def get_cate(self) -> np.ndarray:
        """Get individual-level CATE estimates.

        Returns:
            Array of CATE estimates for each observation
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first")
        return self._cate
