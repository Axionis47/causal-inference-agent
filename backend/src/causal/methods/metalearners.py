"""Meta-Learners for Heterogeneous Treatment Effects (S, T, X Learners)."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .base import BaseCausalMethod, MethodResult


class SLearner(BaseCausalMethod):
    """S-Learner (Single Model Approach).

    Fits a single model with treatment as a feature, then computes
    CATE as the difference in predictions when T=1 vs T=0.

    Simple but may underestimate heterogeneity if treatment effect
    is small relative to main effects.
    """

    METHOD_NAME = "S-Learner"
    ESTIMAND = "CATE"

    def __init__(self, confidence_level: float = 0.95, base_learner: str = "gbm"):
        """Initialize S-Learner.

        Args:
            confidence_level: Confidence level for intervals
            base_learner: Base learner type ('gbm', 'rf', 'linear')
        """
        super().__init__(confidence_level)
        self.base_learner = base_learner
        self._model = None

    def _get_base_model(self):
        """Get base learner model."""
        if self.base_learner == "gbm":
            return GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        elif self.base_learner == "rf":
            return RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        else:
            return LinearRegression()

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> "SLearner":
        """Fit S-Learner model."""
        if not covariates:
            raise ValueError("S-Learner requires covariates")

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + covariates)
        self._T = self._binarize_treatment(df_clean[treatment_col]).values
        self._Y = df_clean[outcome_col].values

        valid_covs = [c for c in covariates if c in df_clean.columns]
        self._X = df_clean[valid_covs].select_dtypes(include=[np.number]).values

        # Include treatment as feature
        X_with_T = np.column_stack([self._X, self._T])

        # Fit single model
        self._model = self._get_base_model()
        self._model.fit(X_with_T, self._Y)

        self._n_treated = int(self._T.sum())
        self._n_control = int(len(self._T) - self._T.sum())
        self._covariates = valid_covs

        self._fitted = True
        return self

    def estimate(self) -> MethodResult:
        """Compute CATE estimates."""
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        # Predict with T=1 and T=0
        X_treated = np.column_stack([self._X, np.ones(len(self._X))])
        X_control = np.column_stack([self._X, np.zeros(len(self._X))])

        mu1 = self._model.predict(X_treated)
        mu0 = self._model.predict(X_control)

        # CATE for each unit
        self._cate = mu1 - mu0

        # ATE is mean of CATEs
        ate = np.mean(self._cate)

        # Bootstrap for SE
        n_bootstrap = 200
        bootstrap_ates = []
        n = len(self._cate)

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            bootstrap_ates.append(np.mean(self._cate[idx]))

        se = np.std(bootstrap_ates)
        ci_lower, ci_upper = np.percentile(bootstrap_ates, [2.5, 97.5])
        p_value = self._compute_p_value(ate, se)

        diagnostics = {
            "cate_mean": float(ate),
            "cate_std": float(np.std(self._cate)),
            "cate_min": float(np.min(self._cate)),
            "cate_max": float(np.max(self._cate)),
            "cate_median": float(np.median(self._cate)),
            "pct_positive": float((self._cate > 0).mean() * 100),
        }

        self._result = MethodResult(
            method=self.METHOD_NAME,
            estimand="ATE",  # Report ATE (mean of CATEs)
            estimate=float(ate),
            std_error=float(se),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(p_value),
            n_treated=self._n_treated,
            n_control=self._n_control,
            assumptions_tested=["conditional_ignorability", "positivity"],
            diagnostics=diagnostics,
            details={
                "base_learner": self.base_learner,
                "n_covariates": len(self._covariates),
            },
        )

        return self._result


class TLearner(BaseCausalMethod):
    """T-Learner (Two Model Approach).

    Fits separate models for treated and control groups,
    then computes CATE as the difference in predictions.

    Better at capturing heterogeneity than S-Learner.
    """

    METHOD_NAME = "T-Learner"
    ESTIMAND = "CATE"

    def __init__(self, confidence_level: float = 0.95, base_learner: str = "gbm"):
        """Initialize T-Learner."""
        super().__init__(confidence_level)
        self.base_learner = base_learner

    def _get_base_model(self):
        """Get base learner model."""
        if self.base_learner == "gbm":
            return GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        elif self.base_learner == "rf":
            return RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        else:
            return LinearRegression()

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> "TLearner":
        """Fit T-Learner models."""
        if not covariates:
            raise ValueError("T-Learner requires covariates")

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + covariates)
        self._T = self._binarize_treatment(df_clean[treatment_col]).values
        self._Y = df_clean[outcome_col].values

        valid_covs = [c for c in covariates if c in df_clean.columns]
        self._X = df_clean[valid_covs].select_dtypes(include=[np.number]).values

        # Split data
        treated_idx = self._T == 1
        control_idx = self._T == 0

        # Fit separate models
        self._model_treated = self._get_base_model()
        self._model_control = self._get_base_model()

        self._model_treated.fit(self._X[treated_idx], self._Y[treated_idx])
        self._model_control.fit(self._X[control_idx], self._Y[control_idx])

        self._n_treated = int(treated_idx.sum())
        self._n_control = int(control_idx.sum())
        self._covariates = valid_covs

        self._fitted = True
        return self

    def estimate(self) -> MethodResult:
        """Compute CATE estimates."""
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        # Predict potential outcomes for all units
        mu1 = self._model_treated.predict(self._X)
        mu0 = self._model_control.predict(self._X)

        # CATE
        self._cate = mu1 - mu0
        ate = np.mean(self._cate)

        # Bootstrap SE
        n_bootstrap = 200
        bootstrap_ates = []
        n = len(self._cate)

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            bootstrap_ates.append(np.mean(self._cate[idx]))

        se = np.std(bootstrap_ates)
        ci_lower, ci_upper = np.percentile(bootstrap_ates, [2.5, 97.5])
        p_value = self._compute_p_value(ate, se)

        diagnostics = {
            "cate_mean": float(ate),
            "cate_std": float(np.std(self._cate)),
            "cate_min": float(np.min(self._cate)),
            "cate_max": float(np.max(self._cate)),
            "pct_positive": float((self._cate > 0).mean() * 100),
        }

        self._result = MethodResult(
            method=self.METHOD_NAME,
            estimand="ATE",
            estimate=float(ate),
            std_error=float(se),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(p_value),
            n_treated=self._n_treated,
            n_control=self._n_control,
            assumptions_tested=["conditional_ignorability", "positivity"],
            diagnostics=diagnostics,
            details={"base_learner": self.base_learner},
        )

        return self._result


class XLearner(BaseCausalMethod):
    """X-Learner.

    Three-stage meta-learner that's particularly effective when
    treatment and control group sizes are imbalanced.

    Stages:
    1. Fit outcome models for treated and control
    2. Impute counterfactual outcomes and compute individual effects
    3. Fit CATE models on imputed effects
    4. Weight CATE predictions by propensity scores
    """

    METHOD_NAME = "X-Learner"
    ESTIMAND = "CATE"

    def __init__(self, confidence_level: float = 0.95, base_learner: str = "gbm"):
        """Initialize X-Learner."""
        super().__init__(confidence_level)
        self.base_learner = base_learner

    def _get_base_model(self):
        """Get base learner model."""
        if self.base_learner == "gbm":
            return GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        elif self.base_learner == "rf":
            return RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        else:
            return LinearRegression()

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> "XLearner":
        """Fit X-Learner models."""
        if not covariates:
            raise ValueError("X-Learner requires covariates")

        from sklearn.linear_model import LogisticRegression

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + covariates)
        self._T = self._binarize_treatment(df_clean[treatment_col]).values
        self._Y = df_clean[outcome_col].values

        valid_covs = [c for c in covariates if c in df_clean.columns]
        self._X = df_clean[valid_covs].select_dtypes(include=[np.number]).values

        treated_idx = self._T == 1
        control_idx = self._T == 0

        X_treated = self._X[treated_idx]
        X_control = self._X[control_idx]
        Y_treated = self._Y[treated_idx]
        Y_control = self._Y[control_idx]

        # Stage 1: Fit outcome models
        model_treated = self._get_base_model()
        model_control = self._get_base_model()
        model_treated.fit(X_treated, Y_treated)
        model_control.fit(X_control, Y_control)

        # Stage 2: Impute treatment effects
        # For treated: D1 = Y1 - mu0(X)
        D1 = Y_treated - model_control.predict(X_treated)
        # For control: D0 = mu1(X) - Y0
        D0 = model_treated.predict(X_control) - Y_control

        # Stage 3: Fit CATE models
        self._tau_model_treated = self._get_base_model()
        self._tau_model_control = self._get_base_model()
        self._tau_model_treated.fit(X_treated, D1)
        self._tau_model_control.fit(X_control, D0)

        # Estimate propensity scores for weighting
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(self._X, self._T)
        self._propensity = ps_model.predict_proba(self._X)[:, 1]
        self._propensity = np.clip(self._propensity, 0.01, 0.99)

        self._n_treated = int(treated_idx.sum())
        self._n_control = int(control_idx.sum())
        self._covariates = valid_covs

        self._fitted = True
        return self

    def estimate(self) -> MethodResult:
        """Compute CATE estimates."""
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        # Predict CATEs from both models
        tau_treated = self._tau_model_treated.predict(self._X)
        tau_control = self._tau_model_control.predict(self._X)

        # Weight by propensity scores
        # tau(x) = g(x) * tau_0(x) + (1 - g(x)) * tau_1(x)
        # where g(x) = P(T=1|X)
        self._cate = self._propensity * tau_control + (1 - self._propensity) * tau_treated

        ate = np.mean(self._cate)

        # Bootstrap SE
        n_bootstrap = 200
        bootstrap_ates = []
        n = len(self._cate)

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            bootstrap_ates.append(np.mean(self._cate[idx]))

        se = np.std(bootstrap_ates)
        ci_lower, ci_upper = np.percentile(bootstrap_ates, [2.5, 97.5])
        p_value = self._compute_p_value(ate, se)

        diagnostics = {
            "cate_mean": float(ate),
            "cate_std": float(np.std(self._cate)),
            "cate_min": float(np.min(self._cate)),
            "cate_max": float(np.max(self._cate)),
            "pct_positive": float((self._cate > 0).mean() * 100),
            "treatment_imbalance": float(abs(self._n_treated - self._n_control) / (self._n_treated + self._n_control)),
        }

        self._result = MethodResult(
            method=self.METHOD_NAME,
            estimand="ATE",
            estimate=float(ate),
            std_error=float(se),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(p_value),
            n_treated=self._n_treated,
            n_control=self._n_control,
            assumptions_tested=["conditional_ignorability", "positivity"],
            diagnostics=diagnostics,
            details={"base_learner": self.base_learner},
        )

        return self._result
