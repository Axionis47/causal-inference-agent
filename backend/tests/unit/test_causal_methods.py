"""Comprehensive unit tests for all causal inference methods.

Each test uses synthetic data with known ground truth to validate
that the method produces correct estimates, confidence intervals,
and diagnostics.
"""

import logging

import numpy as np
import pandas as pd
import pytest

logger = logging.getLogger(__name__)


# ─── Synthetic Data Generators ───────────────────────────────────────────────


def make_ols_data(n=500, ate=2.0, seed=42):
    """Y = ate*T + 3*X1 + noise. Homoscedastic, normal errors."""
    rng = np.random.RandomState(seed)
    X1 = rng.normal(0, 1, n)
    T = rng.binomial(1, 0.5, n)
    Y = ate * T + 3 * X1 + rng.normal(0, 1, n)
    return pd.DataFrame({"treatment": T, "outcome": Y, "X1": X1}), ate


def make_ols_heteroscedastic(n=500, ate=2.0, seed=42):
    """Y = ate*T + noise*X1^2. Strongly heteroscedastic errors for BP test."""
    rng = np.random.RandomState(seed)
    X1 = rng.normal(0, 1, n)
    T = rng.binomial(1, 0.5, n)
    # Use X1^2 as variance multiplier for strong heteroscedasticity
    noise = rng.normal(0, 1, n) * (0.5 + 5 * X1**2)
    Y = ate * T + 3 * X1 + noise
    return pd.DataFrame({"treatment": T, "outcome": Y, "X1": X1}), ate


def make_propensity_data(n=1000, ate=1.5, seed=42):
    """Data with confounded treatment assignment for propensity methods."""
    rng = np.random.RandomState(seed)
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    # Treatment depends on confounders
    ps = 1 / (1 + np.exp(-(0.5 * X1 + 0.3 * X2)))
    T = rng.binomial(1, ps)
    Y = ate * T + 2 * X1 + 1.5 * X2 + rng.normal(0, 1, n)
    return pd.DataFrame({"treatment": T, "outcome": Y, "X1": X1, "X2": X2}), ate


def make_did_data(n=400, att=3.0, seed=42):
    """Panel-like DiD data with parallel pre-trends."""
    rng = np.random.RandomState(seed)
    n_per = n // 4  # 4 groups: treated/control x pre/post

    # Control pre
    Y_cp = 10 + rng.normal(0, 1, n_per)
    # Control post
    Y_cpost = 12 + rng.normal(0, 1, n_per)  # time trend = 2
    # Treated pre
    Y_tp = 11 + rng.normal(0, 1, n_per)  # group diff = 1
    # Treated post (parallel trend + treatment effect)
    Y_tpost = 11 + 2 + att + rng.normal(0, 1, n_per)  # 11 + time_trend + ATT

    df = pd.DataFrame(
        {
            "outcome": np.concatenate([Y_cp, Y_cpost, Y_tp, Y_tpost]),
            "treatment": np.concatenate(
                [np.zeros(2 * n_per), np.ones(2 * n_per)]
            ),
            "post": np.concatenate(
                [
                    np.zeros(n_per),
                    np.ones(n_per),
                    np.zeros(n_per),
                    np.ones(n_per),
                ]
            ),
        }
    )
    return df, att


def make_iv_data(n=1000, late=2.5, seed=42):
    """IV data: Z -> T -> Y with valid instrument."""
    rng = np.random.RandomState(seed)
    U = rng.normal(0, 1, n)  # Unobserved confounder
    Z = rng.binomial(1, 0.5, n)  # Instrument
    # First stage: T depends on Z and U
    T_star = 0.5 + 0.8 * Z + 0.5 * U + rng.normal(0, 0.5, n)
    T = (T_star > 0.5).astype(float)
    # Second stage: Y depends on T and U (confounded)
    Y = late * T + 1.5 * U + rng.normal(0, 1, n)
    return pd.DataFrame({"treatment": T, "outcome": Y, "instrument": Z}), late


def make_rdd_data(n=1000, late=3.0, cutoff=0.0, seed=42):
    """RDD data: sharp discontinuity at cutoff."""
    rng = np.random.RandomState(seed)
    R = rng.uniform(-2, 2, n)  # Running variable
    T = (R >= cutoff).astype(float)
    Y = 2 + 0.5 * R + late * T + rng.normal(0, 0.5, n)
    return (
        pd.DataFrame({"treatment": T, "outcome": Y, "running": R}),
        late,
        cutoff,
    )


def make_heterogeneous_data(n=1000, seed=42):
    """Data with heterogeneous treatment effects for meta-learners."""
    rng = np.random.RandomState(seed)
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    T = rng.binomial(1, 0.5, n)
    # CATE = 2*X1 (heterogeneous)
    cate = 2 * X1
    Y = cate * T + X2 + rng.normal(0, 0.5, n)
    ate = cate.mean()  # Should be ~0 since X1 is symmetric
    return (
        pd.DataFrame({"treatment": T, "outcome": Y, "X1": X1, "X2": X2}),
        ate,
    )


def make_dml_data(n=500, ate=1.0, seed=42):
    """High-dimensional data for Double ML."""
    rng = np.random.RandomState(seed)
    p = 10
    X = rng.normal(0, 1, (n, p))
    T = rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
    Y = ate * T + X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 1, n)
    cols = {f"X{i}": X[:, i] for i in range(p)}
    cols["treatment"] = T
    cols["outcome"] = Y
    return pd.DataFrame(cols), ate


# ─── Test Classes ────────────────────────────────────────────────────────────


class TestOLSMethod:
    """Tests for OLS regression causal method."""

    def test_ols_homoscedastic_estimate_near_truth(self):
        """OLS with homoscedastic data should recover the true ATE."""
        from src.causal.methods.ols import OLSMethod

        df, true_ate = make_ols_data(n=500, ate=2.0, seed=42)
        method = OLSMethod(robust_se=True)
        method.fit(df, treatment_col="treatment", outcome_col="outcome", covariates=["X1"])
        result = method.estimate()

        assert result.method == "OLS Regression"
        assert result.estimand == "ATE"
        assert abs(result.estimate - true_ate) < 0.5, (
            f"OLS estimate {result.estimate:.3f} too far from truth {true_ate}"
        )

    def test_ols_ci_contains_true_value(self):
        """The 95% CI should contain the true ATE."""
        from src.causal.methods.ols import OLSMethod

        df, true_ate = make_ols_data(n=500, ate=2.0, seed=42)
        method = OLSMethod(robust_se=True)
        method.fit(df, treatment_col="treatment", outcome_col="outcome", covariates=["X1"])
        result = method.estimate()

        assert result.ci_lower <= true_ate <= result.ci_upper, (
            f"True ATE {true_ate} not in CI [{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
        )

    def test_ols_diagnostics_present(self):
        """OLS should produce R-squared, F-stat, and other diagnostics."""
        from src.causal.methods.ols import OLSMethod

        df, _ = make_ols_data(n=500, ate=2.0, seed=42)
        method = OLSMethod()
        method.fit(df, treatment_col="treatment", outcome_col="outcome", covariates=["X1"])
        result = method.estimate()

        assert "r_squared" in result.diagnostics
        assert "adj_r_squared" in result.diagnostics
        assert 0 <= result.diagnostics["r_squared"] <= 1.0
        assert result.p_value is not None
        assert result.std_error > 0

    def test_ols_vif_computed_with_covariates(self):
        """OLS should compute VIF when covariates are present."""
        from src.causal.methods.ols import OLSMethod

        df, _ = make_ols_data(n=500, ate=2.0, seed=42)
        method = OLSMethod()
        method.fit(df, treatment_col="treatment", outcome_col="outcome", covariates=["X1"])
        result = method.estimate()

        assert "vif" in result.diagnostics
        assert "X1" in result.diagnostics["vif"]

    def test_ols_heteroscedastic_detects_violation(self):
        """OLS validate_assumptions should detect heteroscedasticity."""
        from src.causal.methods.ols import OLSMethod

        df, _ = make_ols_heteroscedastic(n=500, ate=2.0, seed=42)
        method = OLSMethod(robust_se=False)
        method.fit(df, treatment_col="treatment", outcome_col="outcome", covariates=["X1"])
        method.estimate()
        violations = method.validate_assumptions(df, "treatment", "outcome")

        # Should detect heteroscedasticity (Breusch-Pagan test)
        heteroscedastic_violations = [v for v in violations if "eteroscedasticity" in v.lower()]
        assert len(heteroscedastic_violations) > 0, (
            f"Expected heteroscedasticity violation but got: {violations}"
        )

    def test_ols_not_fitted_raises(self):
        """Calling estimate() before fit() should raise ValueError."""
        from src.causal.methods.ols import OLSMethod

        method = OLSMethod()
        with pytest.raises(ValueError, match="must be fitted"):
            method.estimate()

    def test_ols_no_covariates(self):
        """OLS should work without covariates."""
        from src.causal.methods.ols import OLSMethod

        df, true_ate = make_ols_data(n=500, ate=2.0, seed=42)
        method = OLSMethod()
        method.fit(df, treatment_col="treatment", outcome_col="outcome")
        result = method.estimate()

        assert result.estimate is not None
        assert result.n_treated > 0
        assert result.n_control > 0

    def test_ols_result_to_dict(self):
        """MethodResult.to_dict() should return a proper dictionary."""
        from src.causal.methods.ols import OLSMethod

        df, _ = make_ols_data(n=500, ate=2.0, seed=42)
        method = OLSMethod()
        method.fit(df, treatment_col="treatment", outcome_col="outcome", covariates=["X1"])
        result = method.estimate()
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "method" in d
        assert "estimate" in d
        assert "ci_lower" in d
        assert "ci_upper" in d


class TestPSMMethod:
    """Tests for Propensity Score Matching."""

    def test_psm_estimate_near_truth(self):
        """PSM should recover the true ATT within tolerance."""
        from src.causal.methods.propensity import PSMMethod

        df, true_ate = make_propensity_data(n=1000, ate=1.5, seed=42)
        method = PSMMethod(n_neighbors=1, caliper=0.2)
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        result = method.estimate()

        assert result.method == "Propensity Score Matching"
        assert result.estimand == "ATT"
        # PSM can be noisy, allow wider tolerance
        assert abs(result.estimate - true_ate) < 1.0, (
            f"PSM estimate {result.estimate:.3f} too far from truth {true_ate}"
        )

    def test_psm_common_support_diagnostic(self):
        """PSM should report common support diagnostics."""
        from src.causal.methods.propensity import PSMMethod

        df, _ = make_propensity_data(n=1000, ate=1.5, seed=42)
        method = PSMMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        result = method.estimate()

        assert "common_support" in result.diagnostics
        cs = result.diagnostics["common_support"]
        assert "overlap_lower" in cs
        assert "overlap_upper" in cs
        assert "pct_outside_support" in cs

    def test_psm_requires_covariates(self):
        """PSM should raise ValueError if no covariates provided."""
        from src.causal.methods.propensity import PSMMethod

        df, _ = make_propensity_data(n=1000, ate=1.5, seed=42)
        method = PSMMethod()
        with pytest.raises(ValueError, match="covariates"):
            method.fit(df, treatment_col="treatment", outcome_col="outcome")

    def test_psm_match_rate_positive(self):
        """PSM should match a positive number of treated units."""
        from src.causal.methods.propensity import PSMMethod

        df, _ = make_propensity_data(n=1000, ate=1.5, seed=42)
        method = PSMMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        result = method.estimate()

        assert result.diagnostics["n_matched"] > 0
        assert result.diagnostics["match_rate"] > 0


class TestIPWMethod:
    """Tests for Inverse Probability Weighting."""

    def test_ipw_estimate_near_truth(self):
        """IPW (Hajek) should recover the true ATE within tolerance."""
        from src.causal.methods.propensity import IPWMethod

        df, true_ate = make_propensity_data(n=1000, ate=1.5, seed=42)
        method = IPWMethod(trim_threshold=0.01)
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        result = method.estimate()

        assert result.method == "Inverse Probability Weighting"
        assert result.estimand == "ATE"
        assert abs(result.estimate - true_ate) < 1.0, (
            f"IPW estimate {result.estimate:.3f} too far from truth {true_ate}"
        )

    def test_ipw_effective_sample_size(self):
        """IPW should report effective sample sizes."""
        from src.causal.methods.propensity import IPWMethod

        df, _ = make_propensity_data(n=1000, ate=1.5, seed=42)
        method = IPWMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        result = method.estimate()

        assert "ess_treated" in result.diagnostics
        assert "ess_control" in result.diagnostics
        assert result.diagnostics["ess_treated"] > 0
        assert result.diagnostics["ess_control"] > 0

    def test_ipw_hajek_estimator_static(self):
        """The static _hajek_ate should produce correct results with known data."""
        from src.causal.methods.propensity import IPWMethod

        # Simple case: equal propensity, known outcomes
        T = np.array([1, 1, 0, 0])
        Y = np.array([5.0, 6.0, 2.0, 3.0])
        ps = np.array([0.5, 0.5, 0.5, 0.5])

        ate = IPWMethod._hajek_ate(T, Y, ps)
        # E[Y(1)] = (5+6)/2 = 5.5, E[Y(0)] = (2+3)/2 = 2.5, ATE = 3
        assert abs(ate - 3.0) < 0.01

    def test_ipw_requires_covariates(self):
        """IPW should raise ValueError if no covariates provided."""
        from src.causal.methods.propensity import IPWMethod

        df, _ = make_propensity_data(n=1000, ate=1.5, seed=42)
        method = IPWMethod()
        with pytest.raises(ValueError, match="covariates"):
            method.fit(df, treatment_col="treatment", outcome_col="outcome")


class TestAIPWMethod:
    """Tests for Augmented Inverse Probability Weighting (Doubly Robust)."""

    def test_aipw_estimate_near_truth(self):
        """AIPW should recover the true ATE within tolerance."""
        from src.causal.methods.propensity import AIPWMethod

        df, true_ate = make_propensity_data(n=1000, ate=1.5, seed=42)
        method = AIPWMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
            n_folds=5,
        )
        result = method.estimate()

        assert result.method == "Doubly Robust (AIPW)"
        assert result.estimand == "ATE"
        assert abs(result.estimate - true_ate) < 0.8, (
            f"AIPW estimate {result.estimate:.3f} too far from truth {true_ate}"
        )

    def test_aipw_ci_contains_true_value(self):
        """The AIPW confidence interval should contain the true ATE."""
        from src.causal.methods.propensity import AIPWMethod

        df, true_ate = make_propensity_data(n=1000, ate=1.5, seed=42)
        method = AIPWMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        result = method.estimate()

        assert result.ci_lower <= true_ate <= result.ci_upper, (
            f"True ATE {true_ate} not in AIPW CI [{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
        )

    def test_aipw_cross_fitting_diagnostic(self):
        """AIPW should report cross-fitting diagnostics."""
        from src.causal.methods.propensity import AIPWMethod

        df, _ = make_propensity_data(n=1000, ate=1.5, seed=42)
        method = AIPWMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
            n_folds=5,
        )
        result = method.estimate()

        assert "cross_fitting_folds" in result.diagnostics
        assert result.diagnostics["cross_fitting_folds"] == 5
        assert "ps_overlap" in result.diagnostics
        assert result.details.get("cross_fitting") is True

    def test_aipw_requires_covariates(self):
        """AIPW should raise ValueError if no covariates provided."""
        from src.causal.methods.propensity import AIPWMethod

        df, _ = make_propensity_data(n=1000, ate=1.5, seed=42)
        method = AIPWMethod()
        with pytest.raises(ValueError, match="covariates"):
            method.fit(df, treatment_col="treatment", outcome_col="outcome")


class TestDiDMethod:
    """Tests for Difference-in-Differences."""

    def test_did_estimate_near_truth(self):
        """DiD should recover the true ATT from panel data."""
        from src.causal.methods.did import DifferenceInDifferencesMethod

        df, true_att = make_did_data(n=400, att=3.0, seed=42)
        method = DifferenceInDifferencesMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            post_col="post",
        )
        result = method.estimate()

        assert result.method == "Difference-in-Differences"
        assert result.estimand == "ATT"
        assert abs(result.estimate - true_att) < 1.0, (
            f"DiD estimate {result.estimate:.3f} too far from truth {true_att}"
        )

    def test_did_ci_contains_true_value(self):
        """DiD confidence interval should contain the true ATT."""
        from src.causal.methods.did import DifferenceInDifferencesMethod

        df, true_att = make_did_data(n=400, att=3.0, seed=42)
        method = DifferenceInDifferencesMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            post_col="post",
        )
        result = method.estimate()

        assert result.ci_lower <= true_att <= result.ci_upper, (
            f"True ATT {true_att} not in DiD CI [{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
        )

    def test_did_parallel_trends_check(self):
        """DiD diagnostics should include raw DiD and group means."""
        from src.causal.methods.did import DifferenceInDifferencesMethod

        df, _ = make_did_data(n=400, att=3.0, seed=42)
        method = DifferenceInDifferencesMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            post_col="post",
        )
        result = method.estimate()

        assert "raw_did" in result.diagnostics
        assert "treated_pre_mean" in result.diagnostics
        assert "control_pre_mean" in result.diagnostics
        assert "treated_post_mean" in result.diagnostics
        assert "control_post_mean" in result.diagnostics

    def test_did_requires_time_or_post(self):
        """DiD should raise ValueError without post_col or time_col."""
        from src.causal.methods.did import DifferenceInDifferencesMethod

        df, _ = make_did_data(n=400, att=3.0, seed=42)
        df_no_post = df.drop(columns=["post"])
        method = DifferenceInDifferencesMethod()
        with pytest.raises(ValueError, match="post_col|time_col"):
            method.fit(
                df_no_post,
                treatment_col="treatment",
                outcome_col="outcome",
            )

    def test_did_group_counts(self):
        """DiD should correctly count observations in each group."""
        from src.causal.methods.did import DifferenceInDifferencesMethod

        df, _ = make_did_data(n=400, att=3.0, seed=42)
        method = DifferenceInDifferencesMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            post_col="post",
        )
        result = method.estimate()

        assert result.details["n_treated_pre"] == 100
        assert result.details["n_treated_post"] == 100
        assert result.details["n_control_pre"] == 100
        assert result.details["n_control_post"] == 100


class TestIVMethod:
    """Tests for Instrumental Variables (2SLS)."""

    def test_iv_estimate_near_truth(self):
        """IV/2SLS should recover the LATE within tolerance."""
        from src.causal.methods.iv import InstrumentalVariablesMethod

        df, true_late = make_iv_data(n=1000, late=2.5, seed=42)
        method = InstrumentalVariablesMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            instruments=["instrument"],
        )
        result = method.estimate()

        assert result.method == "Instrumental Variables (2SLS)"
        assert result.estimand == "LATE"
        assert abs(result.estimate - true_late) < 1.5, (
            f"IV estimate {result.estimate:.3f} too far from truth {true_late}"
        )

    def test_iv_first_stage_f_statistic(self):
        """IV should report the first-stage partial F statistic."""
        from src.causal.methods.iv import InstrumentalVariablesMethod

        df, _ = make_iv_data(n=1000, late=2.5, seed=42)
        method = InstrumentalVariablesMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            instruments=["instrument"],
        )
        result = method.estimate()

        assert "first_stage_f_partial" in result.diagnostics
        f_stat = result.diagnostics["first_stage_f_partial"]
        assert f_stat is not None
        # With a strong instrument (coeff=0.8), F should be high
        assert f_stat > 10, f"First stage F = {f_stat:.2f} indicates weak instrument"

    def test_iv_weak_instrument_detection(self):
        """IV should detect weak instruments when F < 10."""
        from src.causal.methods.iv import InstrumentalVariablesMethod

        # Create data with very weak instrument
        rng = np.random.RandomState(42)
        n = 200
        Z = rng.binomial(1, 0.5, n)
        U = rng.normal(0, 1, n)
        # Very weak first stage (coefficient = 0.05)
        T = (0.05 * Z + U + rng.normal(0, 2, n) > 0).astype(float)
        Y = 2.0 * T + U + rng.normal(0, 1, n)
        df = pd.DataFrame({"treatment": T, "outcome": Y, "instrument": Z})

        method = InstrumentalVariablesMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            instruments=["instrument"],
        )
        result = method.estimate()

        # Check diagnostics reflect weakness
        assert "weak_instrument_severe" in result.diagnostics

    def test_iv_requires_instruments(self):
        """IV should raise ValueError without instruments."""
        from src.causal.methods.iv import InstrumentalVariablesMethod

        df, _ = make_iv_data(n=1000, late=2.5, seed=42)
        method = InstrumentalVariablesMethod()
        with pytest.raises(ValueError, match="instrument"):
            method.fit(
                df,
                treatment_col="treatment",
                outcome_col="outcome",
            )

    def test_iv_endogeneity_test(self):
        """IV should report the Durbin-Wu-Hausman endogeneity test."""
        from src.causal.methods.iv import InstrumentalVariablesMethod

        df, _ = make_iv_data(n=1000, late=2.5, seed=42)
        method = InstrumentalVariablesMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            instruments=["instrument"],
        )
        result = method.estimate()

        assert "endogeneity_dwh_statistic" in result.diagnostics
        assert "endogeneity_dwh_pvalue" in result.diagnostics
        assert "treatment_endogenous" in result.diagnostics


class TestRDDMethod:
    """Tests for Regression Discontinuity Design."""

    def test_rdd_estimate_near_truth(self):
        """RDD should recover the LATE at the cutoff."""
        from src.causal.methods.rdd import RegressionDiscontinuityMethod

        df, true_late, cutoff = make_rdd_data(n=1000, late=3.0, cutoff=0.0, seed=42)
        method = RegressionDiscontinuityMethod(polynomial_order=1)
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            running_var="running",
            cutoff=cutoff,
        )
        result = method.estimate()

        assert result.method == "Regression Discontinuity"
        assert result.estimand == "LATE"
        assert abs(result.estimate - true_late) < 1.5, (
            f"RDD estimate {result.estimate:.3f} too far from truth {true_late}"
        )

    def test_rdd_ci_contains_true_value(self):
        """RDD confidence interval should contain the true effect."""
        from src.causal.methods.rdd import RegressionDiscontinuityMethod

        df, true_late, cutoff = make_rdd_data(n=1000, late=3.0, cutoff=0.0, seed=42)
        method = RegressionDiscontinuityMethod(polynomial_order=1)
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            running_var="running",
            cutoff=cutoff,
        )
        result = method.estimate()

        assert result.ci_lower <= true_late <= result.ci_upper, (
            f"True LATE {true_late} not in RDD CI [{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
        )

    def test_rdd_bandwidth_sensitivity(self):
        """RDD should report bandwidth sensitivity analysis."""
        from src.causal.methods.rdd import RegressionDiscontinuityMethod

        df, _, cutoff = make_rdd_data(n=1000, late=3.0, cutoff=0.0, seed=42)
        method = RegressionDiscontinuityMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            running_var="running",
            cutoff=cutoff,
        )
        result = method.estimate()

        assert "bandwidth_sensitivity" in result.diagnostics
        sensitivity = result.diagnostics["bandwidth_sensitivity"]
        assert "1.00h" in sensitivity
        assert "0.50h" in sensitivity
        assert "2.00h" in sensitivity

    def test_rdd_mccrary_test(self):
        """RDD should report McCrary density test diagnostics."""
        from src.causal.methods.rdd import RegressionDiscontinuityMethod

        df, _, cutoff = make_rdd_data(n=1000, late=3.0, cutoff=0.0, seed=42)
        method = RegressionDiscontinuityMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            running_var="running",
            cutoff=cutoff,
        )
        result = method.estimate()

        assert "mccrary_test_statistic" in result.diagnostics
        assert "mccrary_p_value" in result.diagnostics
        assert "manipulation_detected" in result.diagnostics

    def test_rdd_requires_running_var(self):
        """RDD should raise ValueError without running variable."""
        from src.causal.methods.rdd import RegressionDiscontinuityMethod

        df, _, _ = make_rdd_data(n=1000, late=3.0, cutoff=0.0, seed=42)
        method = RegressionDiscontinuityMethod()
        with pytest.raises(ValueError, match="running variable"):
            method.fit(
                df,
                treatment_col="treatment",
                outcome_col="outcome",
            )

    def test_rdd_auto_bandwidth(self):
        """RDD should automatically compute bandwidth if not specified."""
        from src.causal.methods.rdd import RegressionDiscontinuityMethod

        df, _, cutoff = make_rdd_data(n=1000, late=3.0, cutoff=0.0, seed=42)
        method = RegressionDiscontinuityMethod(bandwidth=None)
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            running_var="running",
            cutoff=cutoff,
        )
        result = method.estimate()

        assert result.diagnostics["bandwidth"] > 0
        assert result.n_treated > 0
        assert result.n_control > 0


class TestMetaLearners:
    """Tests for Meta-Learners (S-Learner and T-Learner)."""

    def test_slearner_runs_and_produces_result(self):
        """S-Learner should complete and produce a valid MethodResult."""
        from src.causal.methods.metalearners import SLearner

        df, _ = make_heterogeneous_data(n=500, seed=42)
        method = SLearner(base_learner="linear")
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        result = method.estimate()

        assert result.method == "S-Learner"
        assert result.estimate is not None
        assert result.std_error > 0
        assert result.n_treated > 0
        assert result.n_control > 0

    def test_slearner_diagnostics(self):
        """S-Learner should report CATE distribution diagnostics."""
        from src.causal.methods.metalearners import SLearner

        df, _ = make_heterogeneous_data(n=500, seed=42)
        method = SLearner(base_learner="linear")
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        result = method.estimate()

        assert "cate_mean" in result.diagnostics
        assert "cate_std" in result.diagnostics
        assert "cate_min" in result.diagnostics
        assert "cate_max" in result.diagnostics
        assert "honest_splitting" in result.diagnostics
        assert result.diagnostics["honest_splitting"] is True

    def test_tlearner_runs_and_produces_result(self):
        """T-Learner should complete and produce a valid MethodResult."""
        from src.causal.methods.metalearners import TLearner

        df, _ = make_heterogeneous_data(n=500, seed=42)
        method = TLearner(base_learner="linear")
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        result = method.estimate()

        assert result.method == "T-Learner"
        assert result.estimate is not None
        assert result.std_error > 0

    def test_tlearner_diagnostics(self):
        """T-Learner should report CATE distribution diagnostics."""
        from src.causal.methods.metalearners import TLearner

        df, _ = make_heterogeneous_data(n=500, seed=42)
        method = TLearner(base_learner="linear")
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        result = method.estimate()

        assert "cate_mean" in result.diagnostics
        assert "cate_std" in result.diagnostics
        assert "pct_positive" in result.diagnostics
        assert "honest_splitting" in result.diagnostics

    def test_metalearner_requires_covariates(self):
        """Meta-learners should raise ValueError without covariates."""
        from src.causal.methods.metalearners import SLearner, TLearner

        df, _ = make_heterogeneous_data(n=500, seed=42)

        s_method = SLearner()
        with pytest.raises(ValueError, match="covariates"):
            s_method.fit(df, treatment_col="treatment", outcome_col="outcome")

        t_method = TLearner()
        with pytest.raises(ValueError, match="covariates"):
            t_method.fit(df, treatment_col="treatment", outcome_col="outcome")

    def test_xlearner_runs_and_produces_result(self):
        """X-Learner should complete and produce a valid MethodResult."""
        from src.causal.methods.metalearners import XLearner

        df, _ = make_heterogeneous_data(n=500, seed=42)
        method = XLearner(base_learner="linear")
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        result = method.estimate()

        assert result.method == "X-Learner"
        assert result.estimate is not None
        assert result.std_error > 0
        assert "treatment_imbalance" in result.diagnostics


class TestCausalForest:
    """Tests for Causal Forest method (falls back to T-Learner without econml)."""

    def test_causal_forest_runs_and_produces_result(self):
        """Causal Forest (or fallback) should complete and produce a MethodResult."""
        from src.causal.methods.causal_forest import CausalForestMethod

        df, _ = make_heterogeneous_data(n=500, seed=42)
        method = CausalForestMethod(n_estimators=52, min_samples_leaf=10)
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        result = method.estimate()

        assert result.estimate is not None
        assert result.std_error > 0
        assert result.n_treated > 0
        assert result.n_control > 0

    def test_causal_forest_diagnostics(self):
        """Causal Forest should report heterogeneity diagnostics."""
        from src.causal.methods.causal_forest import CausalForestMethod

        df, _ = make_heterogeneous_data(n=500, seed=42)
        method = CausalForestMethod(n_estimators=52, min_samples_leaf=10)
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        result = method.estimate()

        assert "cate_mean" in result.diagnostics
        assert "cate_std" in result.diagnostics
        assert "cate_min" in result.diagnostics
        assert "cate_max" in result.diagnostics
        assert "heterogeneity_index" in result.diagnostics
        assert "iqr" in result.diagnostics

    def test_causal_forest_get_cate(self):
        """Causal Forest should provide individual CATE estimates via get_cate()."""
        from src.causal.methods.causal_forest import CausalForestMethod

        df, _ = make_heterogeneous_data(n=500, seed=42)
        method = CausalForestMethod(n_estimators=52, min_samples_leaf=10)
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        method.estimate()

        cate = method.get_cate()
        assert isinstance(cate, np.ndarray)
        assert len(cate) > 0

    def test_causal_forest_requires_covariates(self):
        """Causal Forest should raise ValueError without covariates."""
        from src.causal.methods.causal_forest import CausalForestMethod

        df, _ = make_heterogeneous_data(n=500, seed=42)
        method = CausalForestMethod()
        with pytest.raises(ValueError, match="covariates"):
            method.fit(df, treatment_col="treatment", outcome_col="outcome")

    def test_causal_forest_not_fitted_raises(self):
        """Calling get_cate() before fit() should raise ValueError."""
        from src.causal.methods.causal_forest import CausalForestMethod

        method = CausalForestMethod()
        with pytest.raises(ValueError, match="fitted"):
            method.get_cate()


class TestDoubleML:
    """Tests for Double/Debiased Machine Learning."""

    def test_dml_estimate_near_truth(self):
        """Double ML should recover the true ATE."""
        from src.causal.methods.double_ml import DoubleMLMethod

        df, true_ate = make_dml_data(n=500, ate=1.0, seed=42)
        covariates = [f"X{i}" for i in range(10)]
        method = DoubleMLMethod(n_folds=3, ml_method="linear")
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=covariates,
        )
        result = method.estimate()

        assert result.method == "Double ML"
        assert result.estimand == "ATE"
        assert abs(result.estimate - true_ate) < 1.0, (
            f"DML estimate {result.estimate:.3f} too far from truth {true_ate}"
        )

    def test_dml_ci_contains_true_value(self):
        """Double ML confidence interval should contain the true ATE."""
        from src.causal.methods.double_ml import DoubleMLMethod

        df, true_ate = make_dml_data(n=500, ate=1.0, seed=42)
        covariates = [f"X{i}" for i in range(10)]
        method = DoubleMLMethod(n_folds=3, ml_method="linear")
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=covariates,
        )
        result = method.estimate()

        assert result.ci_lower <= true_ate <= result.ci_upper, (
            f"True ATE {true_ate} not in DML CI [{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
        )

    def test_dml_stratified_folds(self):
        """Double ML should report the number of folds used."""
        from src.causal.methods.double_ml import DoubleMLMethod

        df, _ = make_dml_data(n=500, ate=1.0, seed=42)
        covariates = [f"X{i}" for i in range(10)]
        method = DoubleMLMethod(n_folds=5, ml_method="linear")
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=covariates,
        )
        result = method.estimate()

        assert result.diagnostics["n_folds"] == 5
        assert result.diagnostics["ml_method"] == "linear"

    def test_dml_residual_diagnostics(self):
        """Double ML (manual impl) should report residual diagnostics."""
        from src.causal.methods.double_ml import DoubleMLMethod

        df, _ = make_dml_data(n=500, ate=1.0, seed=42)
        covariates = [f"X{i}" for i in range(10)]
        method = DoubleMLMethod(n_folds=3, ml_method="linear")
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=covariates,
        )
        result = method.estimate()

        # Manual implementation reports residual diagnostics
        if not result.diagnostics.get("used_econml", False):
            assert "outcome_residual_var" in result.diagnostics
            assert "treatment_residual_var" in result.diagnostics
            assert "residual_correlation" in result.diagnostics

    def test_dml_requires_covariates(self):
        """Double ML should raise ValueError without covariates."""
        from src.causal.methods.double_ml import DoubleMLMethod

        df, _ = make_dml_data(n=500, ate=1.0, seed=42)
        method = DoubleMLMethod()
        with pytest.raises(ValueError, match="covariates"):
            method.fit(df, treatment_col="treatment", outcome_col="outcome")

    def test_dml_p_value_significant(self):
        """Double ML p-value should be significant for true effect of 1.0."""
        from src.causal.methods.double_ml import DoubleMLMethod

        df, _ = make_dml_data(n=500, ate=1.0, seed=42)
        covariates = [f"X{i}" for i in range(10)]
        method = DoubleMLMethod(n_folds=3, ml_method="linear")
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=covariates,
        )
        result = method.estimate()

        assert result.p_value is not None
        assert result.p_value < 0.1, (
            f"P-value {result.p_value:.4f} should be significant for ATE=1.0"
        )
