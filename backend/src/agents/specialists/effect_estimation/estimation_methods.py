"""Estimation method implementations for treatment effect estimation.

Each method takes (T, Y, X, df) and returns a TreatmentEffectResult or None.
These are pure computational functions — no agent or state logic.
"""

import numpy as np
from scipy import stats

from src.agents.base import TreatmentEffectResult
from src.logging_config.structured import get_logger

from .method_selector import SampleSizeThresholds

logger = get_logger(__name__)


def run_method(
    method: str,
    treatment_var: str,
    outcome_var: str,
    covariates: list[str],
    df,
    current_state=None,
) -> TreatmentEffectResult | None:
    """Run a specific estimation method.

    This delegates to the appropriate estimator function.
    """
    # Filter covariates to numeric-only columns to prevent type errors
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    covariates = [c for c in covariates if c in df.columns and c in numeric_cols]

    # Prepare clean data
    all_cols = [treatment_var, outcome_var] + [c for c in covariates if c in df.columns]
    df_clean = df[all_cols].dropna()

    if len(df_clean) < 50:
        return None

    T = df_clean[treatment_var].values.astype(float)
    Y = df_clean[outcome_var].values.astype(float)
    X = df_clean[covariates].values.astype(float) if covariates else None

    # Ensure binary treatment
    if len(np.unique(T)) > 2:
        T = (T > np.median(T)).astype(int)

    method_map = {
        "ols": estimate_ols,
        "ipw": estimate_ipw,
        "aipw": estimate_aipw,
        "matching": estimate_matching,
        "s_learner": estimate_s_learner,
        "t_learner": estimate_t_learner,
        "x_learner": estimate_x_learner,
        "causal_forest": estimate_causal_forest,
        "double_ml": estimate_double_ml,
        "did": lambda T, Y, X, df: estimate_did(T, Y, X, df, current_state),
        "iv": lambda T, Y, X, df: estimate_iv(T, Y, X, df, current_state),
        "rdd": lambda T, Y, X, df: estimate_rdd(T, Y, X, df, current_state),
    }

    estimator = method_map.get(method)
    if estimator is None:
        return None

    return estimator(T, Y, X, df_clean)


def estimate_ols(T, Y, X, df) -> TreatmentEffectResult:
    """OLS regression estimate."""
    import statsmodels.api as sm

    if X is not None and X.shape[1] > 0:
        design = np.column_stack([np.ones(len(T)), T, X])
    else:
        design = np.column_stack([np.ones(len(T)), T])

    model = sm.OLS(Y, design)
    results = model.fit()

    ate = results.params[1]
    se = results.bse[1]
    ci = results.conf_int()[1]
    pval = results.pvalues[1]

    return TreatmentEffectResult(
        method="OLS Regression",
        estimand="ATE",
        estimate=float(ate),
        std_error=float(se),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        p_value=float(pval),
        assumptions_tested=["Linearity", "No unmeasured confounding"],
        details={"r_squared": float(results.rsquared), "n_obs": int(results.nobs)},
    )


def estimate_ipw(T, Y, X, df) -> TreatmentEffectResult | None:
    """Inverse Probability Weighting estimate."""
    from sklearn.linear_model import LogisticRegression

    if X is None or X.shape[1] == 0:
        return None

    # Estimate propensity scores
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, T)
    ps = ps_model.predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)

    # IPW estimator
    weights_treated = T / ps
    weights_control = (1 - T) / (1 - ps)
    ate = np.mean(weights_treated * Y) - np.mean(weights_control * Y)

    # Bootstrap for SE (stratified to ensure both groups are represented)
    n_bootstrap = 200
    bootstrap_estimates = []
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]
    n_treated_boot = len(treated_idx)
    n_control_boot = len(control_idx)

    for _ in range(n_bootstrap):
        # Stratified bootstrap: sample from each group proportionally
        boot_treated = np.random.choice(treated_idx, size=n_treated_boot, replace=True)
        boot_control = np.random.choice(control_idx, size=n_control_boot, replace=True)
        idx = np.concatenate([boot_treated, boot_control])
        w_t = weights_treated[idx]
        w_c = weights_control[idx]
        y_b = Y[idx]
        bootstrap_estimates.append(np.mean(w_t * y_b) - np.mean(w_c * y_b))

    se = np.std(bootstrap_estimates)
    ci_lower = np.percentile(bootstrap_estimates, 2.5)
    ci_upper = np.percentile(bootstrap_estimates, 97.5)

    return TreatmentEffectResult(
        method="Inverse Probability Weighting",
        estimand="ATE",
        estimate=float(ate),
        std_error=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None,
        assumptions_tested=["Unconfoundedness", "Positivity"],
        details={
            "mean_ps_treated": float(np.mean(ps[T == 1])),
            "mean_ps_control": float(np.mean(ps[T == 0])),
        },
    )


def estimate_aipw(T, Y, X, df) -> TreatmentEffectResult | None:
    """Augmented IPW (Doubly Robust) estimate with regularized models."""
    from sklearn.linear_model import LogisticRegression, Ridge

    if X is None or X.shape[1] == 0:
        return None

    treated_idx = T == 1
    control_idx = T == 0
    n_treated = int(np.sum(treated_idx))
    n_control = int(np.sum(control_idx))

    # Check sample size viability
    viable, reason = SampleSizeThresholds.check_method_viability("aipw", n_treated, n_control)
    if not viable:
        logger.warning("aipw_skipped", reason=reason, n_treated=n_treated, n_control=n_control)
        return None

    # Propensity scores with regularization
    ps_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    ps_model.fit(X, T)
    ps = ps_model.predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)

    # Outcome models with Ridge regularization (more stable for small samples)
    min_arm = min(n_treated, n_control)
    alpha = max(0.1, 10.0 / min_arm)

    mu1_model = Ridge(alpha=alpha)
    mu0_model = Ridge(alpha=alpha)

    mu1_model.fit(X[treated_idx], Y[treated_idx])
    mu0_model.fit(X[control_idx], Y[control_idx])

    mu1 = mu1_model.predict(X)
    mu0 = mu0_model.predict(X)

    # AIPW estimator
    aipw1 = mu1 + T * (Y - mu1) / ps
    aipw0 = mu0 + (1 - T) * (Y - mu0) / (1 - ps)
    ate = np.mean(aipw1 - aipw0)

    # Bootstrap for SE
    n_bootstrap = 200
    bootstrap_estimates = []
    n = len(Y)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        bootstrap_estimates.append(np.mean(aipw1[idx] - aipw0[idx]))

    se = np.std(bootstrap_estimates)
    ci_lower = np.percentile(bootstrap_estimates, 2.5)
    ci_upper = np.percentile(bootstrap_estimates, 97.5)

    # Reliability based on sample size
    if min_arm >= 100:
        reliability = "high"
    elif min_arm >= 50:
        reliability = "medium"
    else:
        reliability = "low"

    return TreatmentEffectResult(
        method="Doubly Robust (AIPW)",
        estimand="ATE",
        estimate=float(ate),
        std_error=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None,
        assumptions_tested=["Unconfoundedness", "Correct propensity OR outcome model"],
        details={
            "is_doubly_robust": True,
            "n_treated": n_treated,
            "n_control": n_control,
            "ridge_alpha": float(alpha),
            "reliability": reliability,
            "mean_ps_treated": float(np.mean(ps[treated_idx])),
            "mean_ps_control": float(np.mean(ps[control_idx])),
        },
    )


def estimate_matching(T, Y, X, df) -> TreatmentEffectResult | None:
    """Propensity Score Matching estimate with k-NN and caliper."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    if X is None or X.shape[1] == 0:
        return None

    treated_idx_orig = np.where(T == 1)[0]
    control_idx_orig = np.where(T == 0)[0]
    n_treated = len(treated_idx_orig)
    n_control = len(control_idx_orig)

    # Check sample size viability
    viable, reason = SampleSizeThresholds.check_method_viability("matching", n_treated, n_control)
    if not viable:
        logger.warning("matching_skipped", reason=reason, n_treated=n_treated, n_control=n_control)
        return None

    # Estimate propensity scores
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, T)
    ps = ps_model.predict_proba(X)[:, 1]

    # Caliper: 0.2 * std of propensity scores (standard practice)
    caliper = 0.2 * np.std(ps)

    # Determine k based on control pool size
    k = min(5, max(1, n_control // n_treated))

    # k-NN matching
    nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
    nn.fit(ps[control_idx_orig].reshape(-1, 1))
    distances, indices = nn.kneighbors(ps[treated_idx_orig].reshape(-1, 1))

    # Apply caliper and compute weighted ATT
    matched_effects = []
    good_matches = 0
    poor_matches = 0

    for i, treat_idx in enumerate(treated_idx_orig):
        matched_distances = distances[i]
        matched_control_indices = control_idx_orig[indices[i]]

        # Apply caliper: only use matches within caliper distance
        within_caliper = matched_distances <= caliper

        if not np.any(within_caliper):
            poor_matches += 1
            best_control = matched_control_indices[0]
            matched_effects.append(Y[treat_idx] - Y[best_control])
        else:
            good_matches += 1
            valid_controls = matched_control_indices[within_caliper]
            valid_distances = matched_distances[within_caliper]

            # Inverse distance weights
            weights = 1.0 / (valid_distances + 1e-6)
            weights = weights / weights.sum()

            control_outcome = np.average(Y[valid_controls], weights=weights)
            matched_effects.append(Y[treat_idx] - control_outcome)

    att = np.mean(matched_effects)
    match_quality = good_matches / n_treated if n_treated > 0 else 0

    # Bootstrap for SE with proper matching
    n_bootstrap = 200
    bootstrap_estimates = []
    for b in range(n_bootstrap):
        boot_idx = np.random.choice(len(matched_effects), size=len(matched_effects), replace=True)
        bootstrap_estimates.append(np.mean([matched_effects[i] for i in boot_idx]))

    se = np.std(bootstrap_estimates)
    ci_lower = np.percentile(bootstrap_estimates, 2.5)
    ci_upper = np.percentile(bootstrap_estimates, 97.5)

    # Reliability based on match quality
    if match_quality >= 0.9:
        reliability = "high"
    elif match_quality >= 0.7:
        reliability = "medium"
    else:
        reliability = "low"

    return TreatmentEffectResult(
        method="Propensity Score Matching (k-NN)",
        estimand="ATT",
        estimate=float(att),
        std_error=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(2 * (1 - stats.norm.cdf(abs(att / se)))) if se > 0 else None,
        assumptions_tested=["Unconfoundedness", "Common support"],
        details={
            "n_treated": n_treated,
            "n_control": n_control,
            "k_neighbors": k,
            "caliper": float(caliper),
            "good_matches": good_matches,
            "poor_matches": poor_matches,
            "match_quality": float(match_quality),
            "reliability": reliability,
            "mean_ps_treated": float(np.mean(ps[treated_idx_orig])),
            "mean_ps_control": float(np.mean(ps[control_idx_orig])),
        },
    )


def estimate_s_learner(T, Y, X, df) -> TreatmentEffectResult | None:
    """S-Learner estimate with adaptive regularization."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    if X is None or X.shape[1] == 0:
        return None

    n_treated = int(np.sum(T == 1))
    n_control = int(np.sum(T == 0))
    n_total = len(T)

    viable, reason = SampleSizeThresholds.check_method_viability("s_learner", n_treated, n_control)
    if not viable:
        logger.warning("s_learner_skipped", reason=reason, n_treated=n_treated, n_control=n_control)
        return None

    X_with_T = np.column_stack([X, T])

    # Adaptive model complexity
    if n_total < 200:
        model = Ridge(alpha=1.0)
        model_type = "ridge"
        reliability = "medium"
    elif n_total < 500:
        model = GradientBoostingRegressor(
            n_estimators=50, max_depth=3,
            min_samples_leaf=max(10, n_total // 50),
            learning_rate=0.1, random_state=42,
        )
        model_type = "gbm_regularized"
        reliability = "medium"
    else:
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=5,
            min_samples_leaf=10, random_state=42,
        )
        model_type = "gbm_full"
        reliability = "high"

    model.fit(X_with_T, Y)

    X_treat = np.column_stack([X, np.ones(len(X))])
    X_control = np.column_stack([X, np.zeros(len(X))])

    y1_pred = model.predict(X_treat)
    y0_pred = model.predict(X_control)
    cate = y1_pred - y0_pred
    ate = np.mean(cate)

    # Bootstrap for SE
    n_bootstrap = 100
    bootstrap_ates = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(Y), size=len(Y), replace=True)
        bootstrap_ates.append(np.mean(cate[idx]))

    se = np.std(bootstrap_ates)
    ci_lower = np.percentile(bootstrap_ates, 2.5)
    ci_upper = np.percentile(bootstrap_ates, 97.5)

    return TreatmentEffectResult(
        method="S-Learner",
        estimand="ATE",
        estimate=float(ate),
        std_error=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None,
        assumptions_tested=["Unconfoundedness"],
        details={
            "cate_std": float(np.std(cate)),
            "model_type": model_type,
            "reliability": reliability,
            "n_total": n_total,
        },
    )


def estimate_t_learner(T, Y, X, df) -> TreatmentEffectResult | None:
    """T-Learner estimate with adaptive regularization based on sample size."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    if X is None or X.shape[1] == 0:
        return None

    treated_idx = T == 1
    control_idx = T == 0
    n_treated = int(np.sum(treated_idx))
    n_control = int(np.sum(control_idx))

    viable, reason = SampleSizeThresholds.check_method_viability("t_learner", n_treated, n_control)
    if not viable:
        logger.warning("t_learner_skipped", reason=reason, n_treated=n_treated, n_control=n_control)
        return None

    min_arm = min(n_treated, n_control)

    if min_arm < 150:
        model_t = Ridge(alpha=1.0)
        model_c = Ridge(alpha=1.0)
        model_type = "ridge"
        reliability = "low"
    elif min_arm < 300:
        model_t = GradientBoostingRegressor(
            n_estimators=50, max_depth=3,
            min_samples_leaf=max(10, min_arm // 20),
            learning_rate=0.1, random_state=42,
        )
        model_c = GradientBoostingRegressor(
            n_estimators=50, max_depth=3,
            min_samples_leaf=max(10, min_arm // 20),
            learning_rate=0.1, random_state=42,
        )
        model_type = "gbm_regularized"
        reliability = "medium"
    else:
        model_t = GradientBoostingRegressor(
            n_estimators=100, max_depth=5,
            min_samples_leaf=10, random_state=42,
        )
        model_c = GradientBoostingRegressor(
            n_estimators=100, max_depth=5,
            min_samples_leaf=10, random_state=42,
        )
        model_type = "gbm_full"
        reliability = "high"

    # Fit models
    model_t.fit(X[treated_idx], Y[treated_idx])
    model_c.fit(X[control_idx], Y[control_idx])

    y1_pred = model_t.predict(X)
    y0_pred = model_c.predict(X)
    cate = y1_pred - y0_pred
    ate = np.mean(cate)

    # Proper bootstrap: re-fit models on each bootstrap sample
    n_bootstrap = 100
    bootstrap_ates = []

    for b in range(n_bootstrap):
        idx_t = np.random.choice(np.where(treated_idx)[0], size=n_treated, replace=True)
        idx_c = np.random.choice(np.where(control_idx)[0], size=n_control, replace=True)

        if min_arm < 150:
            boot_model_t = Ridge(alpha=1.0)
            boot_model_c = Ridge(alpha=1.0)
        elif min_arm < 300:
            boot_model_t = GradientBoostingRegressor(
                n_estimators=30, max_depth=3, min_samples_leaf=max(10, min_arm // 20),
                learning_rate=0.1, random_state=42 + b
            )
            boot_model_c = GradientBoostingRegressor(
                n_estimators=30, max_depth=3, min_samples_leaf=max(10, min_arm // 20),
                learning_rate=0.1, random_state=42 + b
            )
        else:
            boot_model_t = GradientBoostingRegressor(
                n_estimators=50, max_depth=4, min_samples_leaf=10, random_state=42 + b
            )
            boot_model_c = GradientBoostingRegressor(
                n_estimators=50, max_depth=4, min_samples_leaf=10, random_state=42 + b
            )

        try:
            boot_model_t.fit(X[idx_t], Y[idx_t])
            boot_model_c.fit(X[idx_c], Y[idx_c])
            boot_y1 = boot_model_t.predict(X)
            boot_y0 = boot_model_c.predict(X)
            bootstrap_ates.append(np.mean(boot_y1 - boot_y0))
        except Exception:
            continue

    se = np.std(bootstrap_ates) if bootstrap_ates else np.std(cate) / np.sqrt(len(T))
    ci_lower = np.percentile(bootstrap_ates, 2.5) if len(bootstrap_ates) > 10 else ate - 1.96 * se
    ci_upper = np.percentile(bootstrap_ates, 97.5) if len(bootstrap_ates) > 10 else ate + 1.96 * se

    return TreatmentEffectResult(
        method="T-Learner",
        estimand="ATE",
        estimate=float(ate),
        std_error=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None,
        assumptions_tested=["Unconfoundedness"],
        details={
            "cate_std": float(np.std(cate)),
            "n_treated_train": n_treated,
            "n_control_train": n_control,
            "model_type": model_type,
            "reliability": reliability,
            "samples_per_tree_treated": n_treated / (50 if min_arm < 300 else 100),
        },
    )


def estimate_x_learner(T, Y, X, df) -> TreatmentEffectResult | None:
    """X-Learner estimate with adaptive regularization."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    if X is None or X.shape[1] == 0:
        return None

    treated_idx = T == 1
    control_idx = T == 0
    n_treated = int(np.sum(treated_idx))
    n_control = int(np.sum(control_idx))

    viable, reason = SampleSizeThresholds.check_method_viability("x_learner", n_treated, n_control)
    if not viable:
        logger.warning("x_learner_skipped", reason=reason, n_treated=n_treated, n_control=n_control)
        return None

    min_arm = min(n_treated, n_control)

    if min_arm < 150:
        model_t = Ridge(alpha=1.0)
        model_c = Ridge(alpha=1.0)
        tau1_model = Ridge(alpha=1.0)
        tau0_model = Ridge(alpha=1.0)
        model_type = "ridge"
        reliability = "low"
    elif min_arm < 300:
        gbm_params = {
            "n_estimators": 30, "max_depth": 3,
            "min_samples_leaf": max(10, min_arm // 20),
            "learning_rate": 0.1, "random_state": 42,
        }
        model_t = GradientBoostingRegressor(**gbm_params)
        model_c = GradientBoostingRegressor(**gbm_params)
        tau1_model = GradientBoostingRegressor(**gbm_params)
        tau0_model = GradientBoostingRegressor(**gbm_params)
        model_type = "gbm_regularized"
        reliability = "medium"
    else:
        model_t = GradientBoostingRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
        model_c = GradientBoostingRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
        tau1_model = GradientBoostingRegressor(n_estimators=50, max_depth=4, min_samples_leaf=10, random_state=42)
        tau0_model = GradientBoostingRegressor(n_estimators=50, max_depth=4, min_samples_leaf=10, random_state=42)
        model_type = "gbm_full"
        reliability = "high"

    # Step 1: Fit outcome models
    model_t.fit(X[treated_idx], Y[treated_idx])
    model_c.fit(X[control_idx], Y[control_idx])

    # Step 2: Impute treatment effects
    D1 = Y[treated_idx] - model_c.predict(X[treated_idx])
    D0 = model_t.predict(X[control_idx]) - Y[control_idx]

    # Step 3: Fit CATE models
    tau1_model.fit(X[treated_idx], D1)
    tau0_model.fit(X[control_idx], D0)

    # Step 4: Propensity weighting
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X, T)
    ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)

    tau1 = tau1_model.predict(X)
    tau0 = tau0_model.predict(X)
    cate = ps * tau0 + (1 - ps) * tau1
    ate = np.mean(cate)

    # Bootstrap
    n_bootstrap = 100
    bootstrap_ates = []
    for b in range(n_bootstrap):
        idx = np.random.choice(len(Y), size=len(Y), replace=True)
        bootstrap_ates.append(np.mean(cate[idx]))

    se = np.std(bootstrap_ates)
    ci_lower = np.percentile(bootstrap_ates, 2.5)
    ci_upper = np.percentile(bootstrap_ates, 97.5)

    return TreatmentEffectResult(
        method="X-Learner",
        estimand="ATE",
        estimate=float(ate),
        std_error=float(se),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None,
        assumptions_tested=["Unconfoundedness"],
        details={
            "cate_std": float(np.std(cate)),
            "n_treated_train": n_treated,
            "n_control_train": n_control,
            "model_type": model_type,
            "reliability": reliability,
        },
    )


def estimate_causal_forest(T, Y, X, df) -> TreatmentEffectResult | None:
    """Causal Forest estimate using EconML."""
    try:
        from econml.dml import CausalForestDML
    except ImportError:
        return None

    if X is None or X.shape[1] == 0:
        return None

    cf = CausalForestDML(n_estimators=100, min_samples_leaf=10, random_state=42)

    try:
        cf.fit(Y, T, X=X)
        cate = cf.effect(X)
        ate = np.mean(cate)

        ci = cf.effect_interval(X, alpha=0.05)
        ate_ci_lower = np.mean(ci[0])
        ate_ci_upper = np.mean(ci[1])
        se = (ate_ci_upper - ate_ci_lower) / (2 * 1.96)

        return TreatmentEffectResult(
            method="Causal Forest",
            estimand="ATE",
            estimate=float(ate),
            std_error=float(se),
            ci_lower=float(ate_ci_lower),
            ci_upper=float(ate_ci_upper),
            p_value=float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None,
            assumptions_tested=["Unconfoundedness", "Overlap"],
            details={"cate_std": float(np.std(cate))},
        )
    except Exception:
        return None


def estimate_double_ml(T, Y, X, df) -> TreatmentEffectResult | None:
    """Double Machine Learning estimate."""
    try:
        from econml.dml import LinearDML
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    except ImportError:
        return None

    if X is None or X.shape[1] == 0:
        return None

    dml = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=100, random_state=42),
        model_t=GradientBoostingClassifier(n_estimators=100, random_state=42),
        random_state=42,
    )

    try:
        dml.fit(Y, T, X=X)
        ate = dml.ate(X)
        ate_interval = dml.ate_interval(X, alpha=0.05)
        se = (ate_interval[1] - ate_interval[0]) / (2 * 1.96)

        return TreatmentEffectResult(
            method="Double ML",
            estimand="ATE",
            estimate=float(ate),
            std_error=float(se),
            ci_lower=float(ate_interval[0]),
            ci_upper=float(ate_interval[1]),
            p_value=float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None,
            assumptions_tested=["Unconfoundedness", "Overlap"],
            details={"method": "LinearDML"},
        )
    except Exception:
        return None


def estimate_did(T, Y, X, df, current_state=None) -> TreatmentEffectResult | None:
    """Difference-in-Differences estimate."""
    if not current_state or not current_state.data_profile:
        return None

    if not current_state.data_profile.has_time_dimension:
        return None

    time_col = current_state.data_profile.time_column
    if time_col not in df.columns:
        return None

    time_vals = sorted(df[time_col].unique())
    if len(time_vals) < 2:
        return None

    pre_period = time_vals[0]
    post_period = time_vals[-1]

    pre_treated = Y[(T == 1) & (df[time_col] == pre_period)]
    post_treated = Y[(T == 1) & (df[time_col] == post_period)]
    pre_control = Y[(T == 0) & (df[time_col] == pre_period)]
    post_control = Y[(T == 0) & (df[time_col] == post_period)]

    if any(len(g) == 0 for g in [pre_treated, post_treated, pre_control, post_control]):
        return None

    did = (np.mean(post_treated) - np.mean(pre_treated)) - \
          (np.mean(post_control) - np.mean(pre_control))

    var_did = (
        np.var(post_treated) / len(post_treated) +
        np.var(pre_treated) / len(pre_treated) +
        np.var(post_control) / len(post_control) +
        np.var(pre_control) / len(pre_control)
    )
    se = np.sqrt(var_did)

    return TreatmentEffectResult(
        method="Difference-in-Differences",
        estimand="ATT",
        estimate=float(did),
        std_error=float(se),
        ci_lower=float(did - 1.96 * se),
        ci_upper=float(did + 1.96 * se),
        p_value=float(2 * (1 - stats.norm.cdf(abs(did / se)))) if se > 0 else None,
        assumptions_tested=["Parallel trends"],
        details={"time_col": time_col},
    )


def estimate_iv(T, Y, X, df, current_state=None) -> TreatmentEffectResult | None:
    """Instrumental Variables (2SLS) estimate."""
    import statsmodels.api as sm

    if not current_state or not current_state.data_profile:
        return None

    if not current_state.data_profile.potential_instruments:
        return None

    iv_col = current_state.data_profile.potential_instruments[0]
    if iv_col not in df.columns:
        return None

    Z = df[iv_col].values

    # First stage
    Z_design = sm.add_constant(Z)
    first_stage = sm.OLS(T, Z_design).fit()
    f_stat = first_stage.fvalue

    T_hat = first_stage.predict(Z_design)

    # Second stage
    T_hat_design = sm.add_constant(T_hat)
    second_stage = sm.OLS(Y, T_hat_design).fit()

    late = second_stage.params[1]
    se = second_stage.bse[1]
    ci = second_stage.conf_int()[1]

    return TreatmentEffectResult(
        method="Instrumental Variables (2SLS)",
        estimand="LATE",
        estimate=float(late),
        std_error=float(se),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        p_value=float(second_stage.pvalues[1]),
        assumptions_tested=["Relevance", "Exclusion restriction", "Monotonicity"],
        details={"first_stage_f": float(f_stat), "instrument": iv_col},
    )


def estimate_rdd(T, Y, X, df, current_state=None) -> TreatmentEffectResult | None:
    """Regression Discontinuity Design estimate."""
    import statsmodels.api as sm

    if not current_state or not current_state.data_profile:
        return None

    if not current_state.data_profile.discontinuity_candidates:
        return None

    run_col = current_state.data_profile.discontinuity_candidates[0]
    if run_col not in df.columns:
        return None

    R = df[run_col].values
    cutoff = np.median(R)
    R_centered = R - cutoff

    bandwidth = np.std(R) * 0.5
    near_cutoff = np.abs(R_centered) <= bandwidth

    if np.sum(near_cutoff) < 50:
        return None

    R_local = R_centered[near_cutoff]
    Y_local = Y[near_cutoff]
    T_local = (R_local >= 0).astype(int)

    design = np.column_stack([
        np.ones(len(R_local)),
        T_local,
        R_local,
        T_local * R_local,
    ])

    model = sm.OLS(Y_local, design).fit()
    late = model.params[1]
    se = model.bse[1]
    ci = model.conf_int()[1]

    return TreatmentEffectResult(
        method="Regression Discontinuity",
        estimand="LATE",
        estimate=float(late),
        std_error=float(se),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        p_value=float(model.pvalues[1]),
        assumptions_tested=["Continuity at cutoff", "No manipulation"],
        details={"cutoff": float(cutoff), "bandwidth": float(bandwidth), "running_var": run_col},
    )
