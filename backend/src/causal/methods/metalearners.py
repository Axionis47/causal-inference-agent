"""Meta-Learners for Heterogeneous Treatment Effects (S, T, X Learners)."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from .base import BaseCausalMethod, MethodResult


def _select_best_model(X, y, base_learner: str, random_state: int = 42):
    """Select best hyperparameters via 3-fold CV.

    Args:
        X: Feature matrix
        y: Target vector
        base_learner: One of 'gbm', 'rf', 'linear'
        random_state: Random seed

    Returns:
        Fitted model with best hyperparameters
    """
    if base_learner == "linear":
        model = LinearRegression()
        model.fit(X, y)
        return model

    configs = [
        {"n_estimators": 50, "max_depth": 3},
        {"n_estimators": 100, "max_depth": 4},
        {"n_estimators": 200, "max_depth": 5},
    ]

    best_score = -np.inf
    best_config = configs[1]  # default fallback

    for cfg in configs:
        if base_learner == "gbm":
            model = GradientBoostingRegressor(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                random_state=random_state,
            )
        else:  # rf
            model = RandomForestRegressor(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                random_state=random_state,
            )

        try:
            n_cv = min(3, len(y))
            if n_cv < 2:
                score = 0.0
            else:
                scores = cross_val_score(model, X, y, cv=n_cv, scoring="neg_mean_squared_error")
                score = np.mean(scores)
        except Exception:
            score = -np.inf

        if score > best_score:
            best_score = score
            best_config = cfg

    # Fit with best config
    if base_learner == "gbm":
        best_model = GradientBoostingRegressor(
            n_estimators=best_config["n_estimators"],
            max_depth=best_config["max_depth"],
            random_state=random_state,
        )
    else:
        best_model = RandomForestRegressor(
            n_estimators=best_config["n_estimators"],
            max_depth=best_config["max_depth"],
            random_state=random_state,
        )

    best_model.fit(X, y)
    return best_model


class SLearner(BaseCausalMethod):
    """S-Learner (Single Model Approach).

    Fits a single model with treatment as a feature, then computes
    CATE as the difference in predictions when T=1 vs T=0.

    Uses honest sample splitting: fits on first half, estimates CATEs
    on second half to prevent overfitting bias.
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

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> "SLearner":
        """Fit S-Learner model with honest sample splitting."""
        if not covariates:
            raise ValueError("S-Learner requires covariates")

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + covariates)
        T = self._binarize_treatment(df_clean[treatment_col]).values
        Y = df_clean[outcome_col].values

        valid_covs = [c for c in covariates if c in df_clean.columns]
        X = df_clean[valid_covs].select_dtypes(include=[np.number]).values

        # --- Honest sample splitting (50/50) ---
        n = len(Y)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        half = n // 2

        train_idx = perm[:half]
        est_idx = perm[half:]

        X_train, T_train, Y_train = X[train_idx], T[train_idx], Y[train_idx]
        X_est, T_est, Y_est = X[est_idx], T[est_idx], Y[est_idx]

        # Fit model on training half with CV-selected hyperparameters
        X_with_T_train = np.column_stack([X_train, T_train])
        self._model = _select_best_model(X_with_T_train, Y_train, self.base_learner)

        # Store estimation half for CATE estimation
        self._X_est = X_est
        self._T_est = T_est
        self._Y_est = Y_est
        # Store full data for bootstrap
        self._X = X
        self._T = T
        self._Y = Y

        self._n_treated = int(T.sum())
        self._n_control = int(len(T) - T.sum())
        self._covariates = valid_covs

        self._fitted = True
        return self

    def _compute_cate_on_data(self, model, X):
        """Compute CATEs for given X using a fitted model."""
        X_treated = np.column_stack([X, np.ones(len(X))])
        X_control = np.column_stack([X, np.zeros(len(X))])
        return model.predict(X_treated) - model.predict(X_control)

    def estimate(self) -> MethodResult:
        """Compute CATE estimates."""
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        # Estimate CATEs on the estimation (held-out) half
        self._cate = self._compute_cate_on_data(self._model, self._X_est)
        ate = np.mean(self._cate)

        # --- Proper bootstrap SE: resample, refit, recompute ---
        n_bootstrap = 100
        n = len(self._Y)
        bootstrap_ates = []
        rng = np.random.RandomState(123)

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            X_b, T_b, Y_b = self._X[idx], self._T[idx], self._Y[idx]

            # Split bootstrap sample honestly
            half = n // 2
            X_with_T_b = np.column_stack([X_b[:half], T_b[:half]])
            try:
                model_b = _select_best_model(X_with_T_b, Y_b[:half], self.base_learner, random_state=_)
                cate_b = self._compute_cate_on_data(model_b, X_b[half:])
                bootstrap_ates.append(np.mean(cate_b))
            except Exception:
                continue

        if len(bootstrap_ates) < 10:
            se = float(np.std(self._cate) / np.sqrt(len(self._cate)))
        else:
            se = float(np.std(bootstrap_ates))

        ci_lower, ci_upper = np.percentile(bootstrap_ates, [2.5, 97.5]) if len(bootstrap_ates) >= 10 else self._compute_ci(ate, se)
        p_value = self._compute_p_value(ate, se)

        diagnostics = {
            "cate_mean": float(ate),
            "cate_std": float(np.std(self._cate)),
            "cate_min": float(np.min(self._cate)),
            "cate_max": float(np.max(self._cate)),
            "cate_median": float(np.median(self._cate)),
            "pct_positive": float((self._cate > 0).mean() * 100),
            "honest_splitting": True,
            "n_estimation_sample": len(self._X_est),
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

    Uses honest sample splitting: fits on first half, estimates CATEs
    on second half to prevent overfitting bias.
    """

    METHOD_NAME = "T-Learner"
    ESTIMAND = "CATE"

    def __init__(self, confidence_level: float = 0.95, base_learner: str = "gbm"):
        """Initialize T-Learner."""
        super().__init__(confidence_level)
        self.base_learner = base_learner

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> "TLearner":
        """Fit T-Learner models with honest sample splitting."""
        if not covariates:
            raise ValueError("T-Learner requires covariates")

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + covariates)
        T = self._binarize_treatment(df_clean[treatment_col]).values
        Y = df_clean[outcome_col].values

        valid_covs = [c for c in covariates if c in df_clean.columns]
        X = df_clean[valid_covs].select_dtypes(include=[np.number]).values

        # --- Honest sample splitting (50/50) ---
        n = len(Y)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        half = n // 2

        train_idx = perm[:half]
        est_idx = perm[half:]

        X_train, T_train, Y_train = X[train_idx], T[train_idx], Y[train_idx]
        X_est = X[est_idx]

        # Fit separate models on training half
        treated_mask = T_train == 1
        control_mask = T_train == 0

        self._model_treated = _select_best_model(
            X_train[treated_mask], Y_train[treated_mask], self.base_learner
        )
        self._model_control = _select_best_model(
            X_train[control_mask], Y_train[control_mask], self.base_learner
        )

        # Store for estimation and bootstrap
        self._X_est = X_est
        self._X = X
        self._T = T
        self._Y = Y

        self._n_treated = int(T.sum())
        self._n_control = int(len(T) - T.sum())
        self._covariates = valid_covs

        self._fitted = True
        return self

    def estimate(self) -> MethodResult:
        """Compute CATE estimates."""
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        # Predict potential outcomes on estimation half
        mu1 = self._model_treated.predict(self._X_est)
        mu0 = self._model_control.predict(self._X_est)

        # CATE
        self._cate = mu1 - mu0
        ate = np.mean(self._cate)

        # --- Proper bootstrap SE: resample, refit, recompute ---
        n_bootstrap = 100
        n = len(self._Y)
        bootstrap_ates = []
        rng = np.random.RandomState(123)

        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            X_b, T_b, Y_b = self._X[idx], self._T[idx], self._Y[idx]

            half = n // 2
            X_tr, T_tr, Y_tr = X_b[:half], T_b[:half], Y_b[:half]
            X_est_b = X_b[half:]

            treated_mask = T_tr == 1
            control_mask = T_tr == 0

            if treated_mask.sum() < 2 or control_mask.sum() < 2:
                continue

            try:
                m1 = _select_best_model(X_tr[treated_mask], Y_tr[treated_mask], self.base_learner, random_state=b)
                m0 = _select_best_model(X_tr[control_mask], Y_tr[control_mask], self.base_learner, random_state=b)
                cate_b = m1.predict(X_est_b) - m0.predict(X_est_b)
                bootstrap_ates.append(np.mean(cate_b))
            except Exception:
                continue

        if len(bootstrap_ates) < 10:
            se = float(np.std(self._cate) / np.sqrt(len(self._cate)))
        else:
            se = float(np.std(bootstrap_ates))

        ci_lower, ci_upper = np.percentile(bootstrap_ates, [2.5, 97.5]) if len(bootstrap_ates) >= 10 else self._compute_ci(ate, se)
        p_value = self._compute_p_value(ate, se)

        diagnostics = {
            "cate_mean": float(ate),
            "cate_std": float(np.std(self._cate)),
            "cate_min": float(np.min(self._cate)),
            "cate_max": float(np.max(self._cate)),
            "pct_positive": float((self._cate > 0).mean() * 100),
            "honest_splitting": True,
            "n_estimation_sample": len(self._X_est),
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

    Uses honest sample splitting: fits on first half, estimates CATEs
    on second half to prevent overfitting bias.

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

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
        **kwargs: Any,
    ) -> "XLearner":
        """Fit X-Learner models with honest sample splitting."""
        if not covariates:
            raise ValueError("X-Learner requires covariates")

        from sklearn.linear_model import LogisticRegression

        # Prepare data
        df_clean = df.dropna(subset=[treatment_col, outcome_col] + covariates)
        T = self._binarize_treatment(df_clean[treatment_col]).values
        Y = df_clean[outcome_col].values

        valid_covs = [c for c in covariates if c in df_clean.columns]
        X = df_clean[valid_covs].select_dtypes(include=[np.number]).values

        # --- Honest sample splitting (50/50) ---
        n = len(Y)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        half = n // 2

        train_idx = perm[:half]
        est_idx = perm[half:]

        X_train, T_train, Y_train = X[train_idx], T[train_idx], Y[train_idx]
        X_est, T_est = X[est_idx], T[est_idx]

        treated_idx = T_train == 1
        control_idx = T_train == 0

        X_treated = X_train[treated_idx]
        X_control = X_train[control_idx]
        Y_treated = Y_train[treated_idx]
        Y_control = Y_train[control_idx]

        # Stage 1: Fit outcome models with CV-selected hyperparameters
        model_treated = _select_best_model(X_treated, Y_treated, self.base_learner)
        model_control = _select_best_model(X_control, Y_control, self.base_learner)

        # Stage 2: Impute treatment effects
        # For treated: D1 = Y1 - mu0(X)
        D1 = Y_treated - model_control.predict(X_treated)
        # For control: D0 = mu1(X) - Y0
        D0 = model_treated.predict(X_control) - Y_control

        # Stage 3: Fit CATE models
        self._tau_model_treated = _select_best_model(X_treated, D1, self.base_learner)
        self._tau_model_control = _select_best_model(X_control, D0, self.base_learner)

        # Estimate propensity scores for weighting
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, T)
        self._propensity_est = ps_model.predict_proba(X_est)[:, 1]
        self._propensity_est = np.clip(self._propensity_est, 0.01, 0.99)

        # Store for estimation and bootstrap
        self._X_est = X_est
        self._X = X
        self._T = T
        self._Y = Y

        self._n_treated = int(T.sum())
        self._n_control = int(len(T) - T.sum())
        self._covariates = valid_covs

        self._fitted = True
        return self

    def _fit_and_predict_xlearner(self, X, T, Y, X_pred, base_learner, random_state=42):
        """Full X-learner pipeline: fit on (X, T, Y), predict CATEs on X_pred.

        Returns:
            cate array for X_pred
        """
        from sklearn.linear_model import LogisticRegression

        treated_idx = T == 1
        control_idx = T == 0

        if treated_idx.sum() < 2 or control_idx.sum() < 2:
            return None

        X_treated = X[treated_idx]
        X_control = X[control_idx]
        Y_treated = Y[treated_idx]
        Y_control = Y[control_idx]

        model_treated = _select_best_model(X_treated, Y_treated, base_learner, random_state)
        model_control = _select_best_model(X_control, Y_control, base_learner, random_state)

        D1 = Y_treated - model_control.predict(X_treated)
        D0 = model_treated.predict(X_control) - Y_control

        tau_treated = _select_best_model(X_treated, D1, base_learner, random_state)
        tau_control = _select_best_model(X_control, D0, base_learner, random_state)

        ps_model = LogisticRegression(max_iter=1000, random_state=random_state)
        ps_model.fit(X, T)
        propensity = np.clip(ps_model.predict_proba(X_pred)[:, 1], 0.01, 0.99)

        tau_t = tau_treated.predict(X_pred)
        tau_c = tau_control.predict(X_pred)

        # Correct weighting: propensity * tau_control + (1 - propensity) * tau_treated
        cate = propensity * tau_c + (1 - propensity) * tau_t
        return cate

    def estimate(self) -> MethodResult:
        """Compute CATE estimates."""
        if not self._fitted:
            raise ValueError("Model must be fitted before estimating")

        # Predict CATEs from both models on estimation half
        tau_treated = self._tau_model_treated.predict(self._X_est)
        tau_control = self._tau_model_control.predict(self._X_est)

        # Correct X-learner weighting:
        # tau(x) = g(x) * tau_control(x) + (1 - g(x)) * tau_treated(x)
        # where g(x) = P(T=1|X)
        self._cate = self._propensity_est * tau_control + (1 - self._propensity_est) * tau_treated

        ate = np.mean(self._cate)

        # --- Proper bootstrap SE: resample, refit entire pipeline, recompute ---
        n_bootstrap = 100
        n = len(self._Y)
        bootstrap_ates = []
        rng = np.random.RandomState(123)

        for b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            X_b, T_b, Y_b = self._X[idx], self._T[idx], self._Y[idx]

            half = n // 2
            try:
                cate_b = self._fit_and_predict_xlearner(
                    X_b[:half], T_b[:half], Y_b[:half],
                    X_b[half:], self.base_learner, random_state=b,
                )
                if cate_b is not None:
                    bootstrap_ates.append(np.mean(cate_b))
            except Exception:
                continue

        if len(bootstrap_ates) < 10:
            se = float(np.std(self._cate) / np.sqrt(len(self._cate)))
        else:
            se = float(np.std(bootstrap_ates))

        ci_lower, ci_upper = np.percentile(bootstrap_ates, [2.5, 97.5]) if len(bootstrap_ates) >= 10 else self._compute_ci(ate, se)
        p_value = self._compute_p_value(ate, se)

        diagnostics = {
            "cate_mean": float(ate),
            "cate_std": float(np.std(self._cate)),
            "cate_min": float(np.min(self._cate)),
            "cate_max": float(np.max(self._cate)),
            "pct_positive": float((self._cate > 0).mean() * 100),
            "treatment_imbalance": float(abs(self._n_treated - self._n_control) / (self._n_treated + self._n_control)),
            "honest_splitting": True,
            "n_estimation_sample": len(self._X_est),
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
