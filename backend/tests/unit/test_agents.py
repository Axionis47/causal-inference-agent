"""Unit tests for agent functionality."""

import pytest
import numpy as np
import pandas as pd

from src.agents import (
    AnalysisState,
    DataProfile,
    DataProfilerAgent,
    EffectEstimatorAgent,
    OrchestratorAgent,
    SensitivityAnalystAgent,
    TreatmentEffectResult,
)


class TestAnalysisState:
    """Test AnalysisState dataclass."""

    def test_state_initialization(self, analysis_state):
        """Test state initializes with correct defaults."""
        assert analysis_state.job_id == "test-job-123"
        assert analysis_state.treatment_variable == "treatment"
        assert analysis_state.outcome_variable == "outcome"
        assert analysis_state.data_profile is None
        assert len(analysis_state.treatment_effects) == 0

    def test_state_to_dict(self, analysis_state):
        """Test state serialization to dict."""
        state_dict = analysis_state.model_dump()

        assert state_dict["job_id"] == "test-job-123"
        assert "treatment_variable" in state_dict
        assert "outcome_variable" in state_dict

    def test_state_iteration_tracking(self, analysis_state):
        """Test iteration counter."""
        assert analysis_state.iteration_count == 0

        analysis_state.iteration_count = 1
        assert analysis_state.iteration_count == 1


class TestDataProfile:
    """Test DataProfile dataclass."""

    def test_data_profile_creation(self):
        """Test creating a data profile."""
        profile = DataProfile(
            n_samples=1000,
            n_features=10,
            feature_names=["age", "gender"],
            feature_types={"age": "numeric", "gender": "categorical"},
            missing_values={"age": 1},
            numeric_stats={"age": {"mean": 40.0, "std": 10.0, "min": 18.0, "max": 80.0}},
            categorical_stats={"gender": {"male": 500, "female": 500}},
            treatment_candidates=["treatment"],
            outcome_candidates=["outcome"],
        )

        assert profile.n_samples == 1000
        assert profile.n_features == 10
        assert len(profile.feature_types) == 2


class TestTreatmentEffectResult:
    """Test TreatmentEffectResult dataclass."""

    def test_treatment_effect_creation(self):
        """Test creating a treatment effect estimate."""
        effect = TreatmentEffectResult(
            method="OLS Regression",
            estimand="ATE",
            estimate=2.5,
            std_error=0.3,
            ci_lower=1.9,
            ci_upper=3.1,
            p_value=0.001,
        )

        assert effect.estimate == 2.5
        assert effect.ci_lower < effect.estimate < effect.ci_upper

    def test_treatment_effect_significance(self):
        """Test significance based on p-value."""
        significant = TreatmentEffectResult(
            method="OLS",
            estimand="ATE",
            estimate=2.0,
            std_error=0.1,
            ci_lower=1.8,
            ci_upper=2.2,
            p_value=0.001,
        )

        not_significant = TreatmentEffectResult(
            method="OLS",
            estimand="ATE",
            estimate=0.1,
            std_error=0.5,
            ci_lower=-0.9,
            ci_upper=1.1,
            p_value=0.5,
        )

        assert significant.p_value < 0.05
        assert not_significant.p_value >= 0.05


class TestDataProfilerAgent:
    """Test DataProfiler agent."""

    @pytest.fixture
    def profiler(self):
        """Create profiler agent instance."""
        return DataProfilerAgent()

    def test_profiler_initialization(self, profiler):
        """Test profiler initializes correctly."""
        assert profiler.AGENT_NAME == "data_profiler"
        assert profiler.SYSTEM_PROMPT is not None

    def test_profiler_identifies_numeric_columns(self, profiler, sample_dataframe):
        """Test profiler identifies numeric columns."""
        numeric_cols = sample_dataframe.select_dtypes(include=[np.number]).columns
        assert "age" in numeric_cols
        assert "income" in numeric_cols
        assert "treatment" in numeric_cols
        assert "outcome" in numeric_cols

    def test_profiler_identifies_binary_treatment(self, profiler, sample_dataframe):
        """Test profiler identifies binary treatment variable."""
        treatment_col = sample_dataframe["treatment"]
        unique_vals = treatment_col.unique()
        assert len(unique_vals) == 2
        assert set(unique_vals) == {0, 1}


class TestEffectEstimatorAgent:
    """Test EffectEstimator agent."""

    @pytest.fixture
    def estimator(self):
        """Create estimator agent instance."""
        return EffectEstimatorAgent()

    def test_estimator_initialization(self, estimator):
        """Test estimator initializes correctly."""
        assert estimator.AGENT_NAME == "effect_estimator"

    def test_estimator_has_tools(self, estimator):
        """Test estimator has tools defined."""
        assert estimator.TOOLS is not None
        assert len(estimator.TOOLS) > 0

    def test_estimator_has_system_prompt(self, estimator):
        """Test estimator has system prompt defined."""
        assert estimator.SYSTEM_PROMPT is not None
        assert len(estimator.SYSTEM_PROMPT) > 0


class TestOrchestratorAgent:
    """Test Orchestrator agent."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator agent instance."""
        return OrchestratorAgent()

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator.AGENT_NAME == "orchestrator"

    def test_orchestrator_has_system_prompt(self, orchestrator):
        """Test orchestrator has system prompt."""
        assert orchestrator.SYSTEM_PROMPT is not None
        assert len(orchestrator.SYSTEM_PROMPT) > 0

    def test_orchestrator_has_tools(self, orchestrator):
        """Test orchestrator has required tools."""
        assert orchestrator.TOOLS is not None
        assert len(orchestrator.TOOLS) > 0


class TestSensitivityAnalystAgent:
    """Test SensitivityAnalyst agent."""

    @pytest.fixture
    def analyst(self):
        """Create sensitivity analyst instance."""
        return SensitivityAnalystAgent()

    def test_analyst_initialization(self, analyst):
        """Test analyst initializes correctly."""
        assert analyst.AGENT_NAME == "sensitivity_analyst"

    def test_analyst_has_tools(self, analyst):
        """Test analyst has tools defined."""
        assert analyst.TOOLS is not None
        assert len(analyst.TOOLS) > 0

    def test_analyst_has_system_prompt(self, analyst):
        """Test analyst has system prompt defined."""
        assert analyst.SYSTEM_PROMPT is not None
        assert len(analyst.SYSTEM_PROMPT) > 0
