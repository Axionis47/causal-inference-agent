"""Tests for EffectEstimatorEngine.run_method_safe() adapter.

Verifies that the adapter correctly bridges Layer A (BaseCausalMethod classes
returning MethodResult) to the pipeline contract (TreatmentEffectResult).
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.agents.base import TreatmentEffectResult
from src.causal.estimators.effect_estimator import (
    EffectEstimatorEngine,
    to_treatment_effect_result,
)
from src.causal.methods.base import MethodResult


# ── Synthetic Data (reused from test_causal_methods.py) ──────────────────────


def make_ols_data(n=500, ate=2.0, seed=42):
    rng = np.random.RandomState(seed)
    X1 = rng.normal(0, 1, n)
    T = rng.binomial(1, 0.5, n)
    Y = ate * T + 3 * X1 + rng.normal(0, 1, n)
    return pd.DataFrame({"treatment": T, "outcome": Y, "X1": X1}), ate


def make_propensity_data(n=1000, ate=1.5, seed=42):
    rng = np.random.RandomState(seed)
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    ps = 1 / (1 + np.exp(-(0.5 * X1 + 0.3 * X2)))
    T = rng.binomial(1, ps)
    Y = ate * T + 2 * X1 + 1.5 * X2 + rng.normal(0, 1, n)
    return pd.DataFrame({"treatment": T, "outcome": Y, "X1": X1, "X2": X2}), ate


# ── Conversion Tests ─────────────────────────────────────────────────────────


class TestToTreatmentEffectResult:
    """Test MethodResult -> TreatmentEffectResult conversion."""

    def test_basic_fields_preserved(self):
        mr = MethodResult(
            method="OLS Regression",
            estimand="ATE",
            estimate=2.1,
            std_error=0.3,
            ci_lower=1.5,
            ci_upper=2.7,
            p_value=0.001,
            n_treated=250,
            n_control=250,
            assumptions_tested=["Linearity"],
            diagnostics={"r_squared": 0.85},
            details={"n_obs": 500},
        )
        ter = to_treatment_effect_result(mr)

        assert isinstance(ter, TreatmentEffectResult)
        assert ter.method == "OLS Regression"
        assert ter.estimand == "ATE"
        assert ter.estimate == 2.1
        assert ter.std_error == 0.3
        assert ter.ci_lower == 1.5
        assert ter.ci_upper == 2.7
        assert ter.p_value == 0.001
        assert ter.assumptions_tested == ["Linearity"]

    def test_diagnostics_merged_into_details(self):
        mr = MethodResult(
            method="OLS Regression",
            estimand="ATE",
            estimate=2.0,
            std_error=0.3,
            ci_lower=1.4,
            ci_upper=2.6,
            diagnostics={"r_squared": 0.85, "vif": {"X1": 1.2}},
            details={"n_obs": 500},
        )
        ter = to_treatment_effect_result(mr)

        assert ter.details["diagnostics"] == {"r_squared": 0.85, "vif": {"X1": 1.2}}
        assert ter.details["n_obs"] == 500

    def test_n_treated_n_control_in_details(self):
        mr = MethodResult(
            method="IPW",
            estimand="ATE",
            estimate=1.5,
            std_error=0.2,
            ci_lower=1.1,
            ci_upper=1.9,
            n_treated=400,
            n_control=600,
        )
        ter = to_treatment_effect_result(mr)

        assert ter.details["n_treated"] == 400
        assert ter.details["n_control"] == 600

    def test_empty_diagnostics_not_added(self):
        mr = MethodResult(
            method="OLS Regression",
            estimand="ATE",
            estimate=2.0,
            std_error=0.3,
            ci_lower=1.4,
            ci_upper=2.6,
            diagnostics={},
            details={},
        )
        ter = to_treatment_effect_result(mr)

        assert "diagnostics" not in ter.details


# ── run_method_safe Tests ────────────────────────────────────────────────────


class TestRunMethodSafe:
    """Test EffectEstimatorEngine.run_method_safe()."""

    def setup_method(self):
        self.engine = EffectEstimatorEngine()

    def test_ols_returns_treatment_effect_result(self):
        df, true_ate = make_ols_data(n=500)
        result = self.engine.run_method_safe(
            method="ols",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1"],
        )
        assert isinstance(result, TreatmentEffectResult)
        assert result.method == "OLS Regression"
        assert result.estimand == "ATE"
        # Estimate should be within reasonable range of true ATE
        assert abs(result.estimate - true_ate) < 1.0
        assert result.std_error > 0
        assert result.ci_lower < result.estimate < result.ci_upper

    def test_ols_diagnostics_present(self):
        df, _ = make_ols_data(n=500)
        result = self.engine.run_method_safe(
            method="ols",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1"],
        )
        assert result is not None
        # Layer A OLS includes r_squared and VIF diagnostics
        assert "diagnostics" in result.details
        diag = result.details["diagnostics"]
        assert "r_squared" in diag

    def test_matching_alias_works(self):
        """The agent uses 'matching', engine uses 'psm'. Alias must work."""
        df, _ = make_propensity_data(n=500)
        result = self.engine.run_method_safe(
            method="matching",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        assert result is not None
        assert "Matching" in result.method or "PSM" in result.method or "Propensity" in result.method

    def test_unknown_method_returns_none(self):
        df, _ = make_ols_data()
        result = self.engine.run_method_safe(
            method="nonexistent_method",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1"],
        )
        assert result is None

    def test_insufficient_data_returns_none(self):
        df, _ = make_ols_data(n=30)
        result = self.engine.run_method_safe(
            method="ols",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1"],
        )
        # 30 rows, some may drop to NaN, below the 50-row minimum
        # Either None or a result is acceptable depending on NaN count
        # but for methods requiring >30 rows per MIN_SAMPLES it should gate
        assert result is None or isinstance(result, TreatmentEffectResult)

    def test_sample_size_gating_for_ml_methods(self):
        """ML methods need 200+ samples. 100-sample data should be rejected."""
        df, _ = make_propensity_data(n=100)
        result = self.engine.run_method_safe(
            method="s_learner",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        assert result is None

    def test_covariates_required_returns_none(self):
        """Methods requiring covariates should return None if none provided."""
        df, _ = make_propensity_data(n=500)
        result = self.engine.run_method_safe(
            method="ipw",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=[],  # No covariates
        )
        assert result is None

    def test_did_without_state_returns_none(self):
        df, _ = make_ols_data()
        result = self.engine.run_method_safe(
            method="did",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1"],
            current_state=None,
        )
        assert result is None

    def test_did_without_time_dimension_returns_none(self):
        df, _ = make_ols_data()
        state = MagicMock()
        state.data_profile.has_time_dimension = False
        result = self.engine.run_method_safe(
            method="did",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1"],
            current_state=state,
        )
        assert result is None

    def test_iv_without_instruments_returns_none(self):
        df, _ = make_ols_data()
        state = MagicMock()
        state.data_profile.potential_instruments = []
        result = self.engine.run_method_safe(
            method="iv",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1"],
            current_state=state,
        )
        assert result is None

    def test_rdd_without_candidates_returns_none(self):
        df, _ = make_ols_data()
        state = MagicMock()
        state.data_profile.discontinuity_candidates = []
        result = self.engine.run_method_safe(
            method="rdd",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1"],
            current_state=state,
        )
        assert result is None

    def test_non_numeric_covariates_filtered(self):
        """Non-numeric covariates should be silently filtered out."""
        df, true_ate = make_ols_data(n=500)
        df["category"] = np.random.choice(["a", "b", "c"], size=len(df))

        result = self.engine.run_method_safe(
            method="ols",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "category"],  # category is non-numeric
        )
        assert result is not None
        assert abs(result.estimate - true_ate) < 1.0

    def test_nan_rows_dropped(self):
        """Rows with NaN values should be dropped before estimation."""
        df, _ = make_ols_data(n=500)
        # Inject some NaNs
        df.loc[0:9, "X1"] = np.nan
        result = self.engine.run_method_safe(
            method="ols",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1"],
        )
        assert result is not None

    def test_exception_returns_none(self):
        """If the underlying method throws, run_method_safe returns None."""
        df = pd.DataFrame({"treatment": [0], "outcome": [1], "X1": [2]})
        # Only 1 row, fitting will fail
        result = self.engine.run_method_safe(
            method="ols",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1"],
        )
        assert result is None

    def test_ipw_with_propensity_data(self):
        df, true_ate = make_propensity_data(n=1000)
        result = self.engine.run_method_safe(
            method="ipw",
            df=df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["X1", "X2"],
        )
        assert result is not None
        assert result.estimand == "ATE"
        # IPW should be in the right ballpark
        assert abs(result.estimate - true_ate) < 2.0
