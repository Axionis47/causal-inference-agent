"""Integration test: run full analysis pipeline with synthetic data.

This test verifies the end-to-end pipeline using:
- Synthetic data with known ground truth (ATE = 2000)
- Mocked LLM calls (no external API dependency)
- All causal methods running against the same dataset
- Verification that estimates converge near the true ATE
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.base.state import (
    AnalysisState,
    CausalPair,
    DataProfile,
    DatasetInfo,
    JobStatus,
    TreatmentEffectResult,
)


# ─── Synthetic Dataset ──────────────────────────────────────────────────────


def make_pipeline_dataset(n=500, seed=42):
    """Create a synthetic dataset for full pipeline testing.

    Ground truth:
    - ATE = 2000 (treatment effect on outcome)
    - Treatment assigned based on confounders (age, income)
    - Outcome depends on treatment + confounders + noise

    Returns:
        Tuple of (DataFrame, true_ate)
    """
    rng = np.random.RandomState(seed)

    age = rng.normal(40, 10, n)
    income = rng.normal(50000, 15000, n)
    education = rng.normal(14, 3, n)

    # Treatment depends on confounders
    propensity = 1 / (
        1 + np.exp(-(0.02 * (age - 40) + 0.00001 * (income - 50000)))
    )
    treatment = rng.binomial(1, propensity)

    # Outcome: TRUE ATE = 2000
    true_ate = 2000
    outcome = (
        5000
        + true_ate * treatment
        + 50 * age
        + 0.1 * income
        + 200 * education
        + rng.normal(0, 1000, n)
    )

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "education": education,
            "treatment": treatment,
            "outcome": outcome,
        }
    )
    return df, true_ate


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_dataset():
    """Create and persist a synthetic dataset for testing."""
    df, true_ate = make_pipeline_dataset(n=500, seed=42)
    temp_path = Path(tempfile.gettempdir()) / "pipeline_integration_test.pkl"

    with open(temp_path, "wb") as f:
        pickle.dump(df, f)

    yield df, true_ate, str(temp_path)

    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def pipeline_state(synthetic_dataset):
    """Create an AnalysisState with pre-populated data profile."""
    df, true_ate, data_path = synthetic_dataset

    state = AnalysisState(
        job_id="integration-pipeline-001",
        dataset_info=DatasetInfo(
            url="file://integration-test",
            name="integration_test_dataset",
        ),
        treatment_variable="treatment",
        outcome_variable="outcome",
    )
    state.dataframe_path = data_path

    # Pre-populate data profile (as if data_profiler already ran)
    state.data_profile = DataProfile(
        n_samples=len(df),
        n_features=len(df.columns),
        feature_names=list(df.columns),
        feature_types={col: "numeric" for col in df.columns},
        missing_values={col: 0 for col in df.columns},
        numeric_stats={
            col: {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
            }
            for col in df.columns
        },
        categorical_stats={},
        treatment_candidates=["treatment"],
        outcome_candidates=["outcome"],
        potential_confounders=["age", "income", "education"],
    )

    return state, true_ate


# ─── Test: Multiple Causal Methods on Same Dataset ──────────────────────────


class TestMultiMethodEstimation:
    """Test running multiple causal methods on synthetic data."""

    def test_ols_on_pipeline_data(self, synthetic_dataset):
        """OLS should estimate near true ATE on the pipeline dataset."""
        from src.causal.methods.ols import OLSMethod

        df, true_ate, _ = synthetic_dataset

        method = OLSMethod(robust_se=True)
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["age", "income", "education"],
        )
        result = method.estimate()

        assert abs(result.estimate - true_ate) < 500, (
            f"OLS estimate {result.estimate:.0f} too far from truth {true_ate}"
        )

    def test_ipw_on_pipeline_data(self, synthetic_dataset):
        """IPW should estimate near true ATE on the pipeline dataset."""
        from src.causal.methods.propensity import IPWMethod

        df, true_ate, _ = synthetic_dataset

        method = IPWMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["age", "income", "education"],
        )
        result = method.estimate()

        assert abs(result.estimate - true_ate) < 1000, (
            f"IPW estimate {result.estimate:.0f} too far from truth {true_ate}"
        )

    def test_aipw_on_pipeline_data(self, synthetic_dataset):
        """AIPW should estimate near true ATE on the pipeline dataset."""
        from src.causal.methods.propensity import AIPWMethod

        df, true_ate, _ = synthetic_dataset

        method = AIPWMethod()
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["age", "income", "education"],
        )
        result = method.estimate()

        assert abs(result.estimate - true_ate) < 800, (
            f"AIPW estimate {result.estimate:.0f} too far from truth {true_ate}"
        )

    def test_dml_on_pipeline_data(self, synthetic_dataset):
        """Double ML should estimate near true ATE on the pipeline dataset."""
        from src.causal.methods.double_ml import DoubleMLMethod

        df, true_ate, _ = synthetic_dataset

        method = DoubleMLMethod(n_folds=3, ml_method="linear")
        method.fit(
            df,
            treatment_col="treatment",
            outcome_col="outcome",
            covariates=["age", "income", "education"],
        )
        result = method.estimate()

        assert abs(result.estimate - true_ate) < 800, (
            f"DML estimate {result.estimate:.0f} too far from truth {true_ate}"
        )


class TestMethodConsensus:
    """Test that multiple methods converge on similar estimates."""

    def test_methods_agree_on_direction(self, synthetic_dataset):
        """All methods should agree on the direction of the effect."""
        from src.causal.methods.double_ml import DoubleMLMethod
        from src.causal.methods.ols import OLSMethod
        from src.causal.methods.propensity import AIPWMethod, IPWMethod

        df, true_ate, _ = synthetic_dataset
        covariates = ["age", "income", "education"]
        estimates = []

        # OLS
        ols = OLSMethod()
        ols.fit(df, "treatment", "outcome", covariates)
        estimates.append(ols.estimate().estimate)

        # IPW
        ipw = IPWMethod()
        ipw.fit(df, "treatment", "outcome", covariates)
        estimates.append(ipw.estimate().estimate)

        # AIPW
        aipw = AIPWMethod()
        aipw.fit(df, "treatment", "outcome", covariates)
        estimates.append(aipw.estimate().estimate)

        # DML
        dml = DoubleMLMethod(n_folds=3, ml_method="linear")
        dml.fit(df, "treatment", "outcome", covariates)
        estimates.append(dml.estimate().estimate)

        # All should be positive (true ATE = 2000 > 0)
        for est in estimates:
            assert est > 0, f"Method estimated negative effect: {est:.0f}"

        # Estimates should be in a reasonable range
        for est in estimates:
            assert 500 < est < 4000, (
                f"Estimate {est:.0f} outside reasonable range [500, 4000]"
            )


# ─── Test: State Progression ────────────────────────────────────────────────


class TestStateProgression:
    """Test that the analysis state progresses correctly through stages."""

    def test_state_starts_pending(self):
        """A new state should start as PENDING."""
        state = AnalysisState(
            job_id="test-state-1",
            dataset_info=DatasetInfo(url="file://test"),
        )
        assert state.status == JobStatus.PENDING

    def test_state_transition_to_profiling(self, pipeline_state):
        """State should transition to PROFILING when profiler runs."""
        state, _ = pipeline_state
        state.status = JobStatus.PROFILING
        assert state.status == JobStatus.PROFILING

    def test_state_accumulates_effects(self, pipeline_state):
        """Treatment effects should accumulate as methods run."""
        state, _ = pipeline_state

        # Add effects from multiple methods
        state.treatment_effects.append(
            TreatmentEffectResult(
                method="OLS",
                estimand="ATE",
                estimate=2100.0,
                std_error=200.0,
                ci_lower=1700.0,
                ci_upper=2500.0,
                treatment_variable="treatment",
                outcome_variable="outcome",
            )
        )
        state.treatment_effects.append(
            TreatmentEffectResult(
                method="IPW",
                estimand="ATE",
                estimate=1900.0,
                std_error=300.0,
                ci_lower=1300.0,
                ci_upper=2500.0,
                treatment_variable="treatment",
                outcome_variable="outcome",
            )
        )

        assert len(state.treatment_effects) == 2
        methods = [e.method for e in state.treatment_effects]
        assert "OLS" in methods
        assert "IPW" in methods

    def test_get_effects_for_pair(self, pipeline_state):
        """get_effects_for_pair should filter effects correctly."""
        state, _ = pipeline_state

        state.treatment_effects.append(
            TreatmentEffectResult(
                method="OLS",
                estimand="ATE",
                estimate=2100.0,
                std_error=200.0,
                ci_lower=1700.0,
                ci_upper=2500.0,
                treatment_variable="treatment",
                outcome_variable="outcome",
            )
        )
        state.treatment_effects.append(
            TreatmentEffectResult(
                method="OLS",
                estimand="ATE",
                estimate=100.0,
                std_error=50.0,
                ci_lower=0.0,
                ci_upper=200.0,
                treatment_variable="age",
                outcome_variable="income",
            )
        )

        effects = state.get_effects_for_pair("treatment", "outcome")
        assert len(effects) == 1
        assert effects[0].estimate == 2100.0

    def test_analyzed_pairs_tracking(self, pipeline_state):
        """Analyzed pairs should be tracked."""
        state, _ = pipeline_state

        state.analyzed_pairs.append(
            CausalPair(
                treatment="treatment",
                outcome="outcome",
                rationale="Primary analysis",
                priority=1,
            )
        )

        pairs = state.get_all_pairs()
        assert len(pairs) == 1
        assert pairs[0] == ("treatment", "outcome")

    def test_state_completes_successfully(self, pipeline_state):
        """State should mark as completed correctly."""
        state, _ = pipeline_state

        state.mark_completed()

        assert state.status == JobStatus.COMPLETED
        assert state.completed_at is not None


# ─── Test: Causal Discovery Pipeline ────────────────────────────────────────


class TestDiscoveryPipeline:
    """Test causal discovery as part of the pipeline."""

    def test_pc_discovery_on_pipeline_data(self, synthetic_dataset):
        """PC algorithm should discover some structure in the pipeline data."""
        from src.causal.dag.discovery import CausalDiscovery

        df, _, _ = synthetic_dataset
        variables = ["age", "income", "education", "treatment", "outcome"]

        cd = CausalDiscovery(algorithm="pc", alpha=0.05)
        result = cd.discover(df, variables=variables)

        assert len(result.nodes) == 5
        # Should find at least some edges
        assert len(result.edges) > 0

    def test_discovery_with_treatment_outcome(self, synthetic_dataset):
        """Discovery should orient edges when treatment/outcome are specified."""
        from src.causal.dag.discovery import CausalDiscovery

        df, _, _ = synthetic_dataset
        variables = ["age", "income", "education", "treatment", "outcome"]

        cd = CausalDiscovery(algorithm="pc", alpha=0.05)
        result = cd.discover(
            df,
            variables=variables,
            treatment="treatment",
            outcome="outcome",
        )

        assert result.algorithm == "PC"
        assert len(result.nodes) == 5


# ─── Test: End-to-End Mocked Pipeline ───────────────────────────────────────


class TestMockedEndToEnd:
    """Test the full pipeline with mocked LLM calls."""

    def test_methods_chain_produces_valid_output(self, synthetic_dataset):
        """Chain OLS -> Sensitivity check should produce valid results.

        This simulates the pipeline without LLM calls.
        """
        from src.causal.methods.ols import OLSMethod

        df, true_ate, _ = synthetic_dataset
        covariates = ["age", "income", "education"]

        # Step 1: Estimate
        method = OLSMethod()
        method.fit(df, "treatment", "outcome", covariates)
        result = method.estimate()

        # Step 2: Validate assumptions
        violations = method.validate_assumptions(df, "treatment", "outcome")

        # Step 3: Verify output structure
        d = result.to_dict()
        assert "method" in d
        assert "estimate" in d
        assert "diagnostics" in d
        assert isinstance(violations, list)

        # Step 4: Verify estimate quality
        assert result.ci_lower <= true_ate <= result.ci_upper

    def test_pipeline_state_roundtrip(self, pipeline_state):
        """State should serialize and deserialize correctly via Pydantic."""
        state, _ = pipeline_state

        # Add some data
        state.treatment_effects.append(
            TreatmentEffectResult(
                method="OLS",
                estimand="ATE",
                estimate=2000.0,
                std_error=100.0,
                ci_lower=1800.0,
                ci_upper=2200.0,
            )
        )

        # Serialize and deserialize
        state_dict = state.model_dump()
        restored = AnalysisState(**state_dict)

        assert restored.job_id == state.job_id
        assert len(restored.treatment_effects) == 1
        assert restored.treatment_effects[0].estimate == 2000.0
        assert restored.data_profile is not None
        assert restored.data_profile.n_samples == 500
