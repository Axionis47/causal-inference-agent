"""Unit tests for DataProfilerAgent (ReAct-based)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock

from src.agents.base import AnalysisState, DatasetInfo, ToolResultStatus
from src.agents.specialists.data_profiler import DataProfilerAgent


@pytest.fixture
def agent():
    """Create agent instance."""
    return DataProfilerAgent()


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        "treat": np.random.binomial(1, 0.3, n),  # Binary treatment (30% treated)
        "age": np.random.normal(40, 10, n),
        "income": np.random.normal(50000, 15000, n),
        "education": np.random.randint(10, 20, n),
        "outcome": np.random.normal(100, 20, n),
        "gender": np.random.choice(["M", "F"], n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
    })


@pytest.fixture
def state_with_dataframe(sample_dataframe, tmp_path):
    """Create state with a loaded dataframe."""
    # Save dataframe to temp path
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)

    return AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(
            url="https://kaggle.com/test/dataset",
            name="test_dataset",
            local_path=str(csv_path),
        ),
    )


class TestInitialization:
    """Test agent initialization."""

    def test_agent_name(self, agent):
        """Test agent has correct name."""
        assert agent.AGENT_NAME == "data_profiler"

    def test_max_steps(self, agent):
        """Test agent has reasonable max steps."""
        assert agent.MAX_STEPS == 15

    def test_tools_registered(self, agent):
        """Test that profiling tools are registered."""
        tool_names = list(agent._tools.keys())

        expected_tools = [
            # Profiling tools
            "get_dataset_overview",
            "analyze_column",
            "check_treatment_balance",
            "check_column_relationship",
            "check_time_dimension",
            "check_discontinuity_candidates",
            "finalize_profile",
            # Context tools from mixin
            "ask_domain_knowledge",
            "get_column_info",
            "get_dataset_summary",
            "list_columns",
            # Built-in ReAct tools
            "finish",
            "reflect",
        ]

        for tool in expected_tools:
            assert tool in tool_names, f"Tool '{tool}' not registered"

    def test_has_context_tools(self, agent):
        """Test that context tools mixin is properly integrated."""
        # Context tools should be available
        assert "ask_domain_knowledge" in agent._tools
        assert "get_column_info" in agent._tools


class TestToolGetOverview:
    """Test get_dataset_overview tool."""

    @pytest.mark.asyncio
    async def test_returns_overview(self, agent, sample_dataframe, state_with_dataframe):
        """Test getting dataset overview."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_get_overview(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_samples"] == 100
        assert result.output["n_features"] == 7
        assert "binary_columns" in result.output
        assert "numeric_columns" in result.output
        assert "suggestions" in result.output

    @pytest.mark.asyncio
    async def test_identifies_binary_columns(self, agent, sample_dataframe, state_with_dataframe):
        """Test that binary columns are identified."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_get_overview(state_with_dataframe)

        assert "treat" in result.output["binary_columns"]

    @pytest.mark.asyncio
    async def test_error_when_no_dataframe(self, agent, state_with_dataframe):
        """Test error when dataframe not loaded."""
        agent._df = None
        agent._profile = None

        result = await agent._tool_get_overview(state_with_dataframe)

        assert result.status == ToolResultStatus.ERROR
        assert "not loaded" in result.error


class TestToolAnalyzeColumn:
    """Test analyze_column tool."""

    @pytest.mark.asyncio
    async def test_analyzes_binary_column(self, agent, sample_dataframe, state_with_dataframe):
        """Test analyzing a binary column."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_analyze_column(state_with_dataframe, column="treat")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["column"] == "treat"
        assert result.output["type"] == "binary"
        assert "value_distribution" in result.output
        assert "treatment_suitability" in result.output

    @pytest.mark.asyncio
    async def test_analyzes_numeric_column(self, agent, sample_dataframe, state_with_dataframe):
        """Test analyzing a numeric column."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_analyze_column(state_with_dataframe, column="income")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["column"] == "income"
        assert result.output["type"] == "numeric"
        assert "statistics" in result.output
        assert "mean" in result.output["statistics"]
        assert "std" in result.output["statistics"]
        assert result.output["has_variance"] == True

    @pytest.mark.asyncio
    async def test_column_not_found(self, agent, sample_dataframe, state_with_dataframe):
        """Test error when column not found."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_analyze_column(state_with_dataframe, column="nonexistent")

        assert result.status == ToolResultStatus.ERROR
        assert "not found" in result.error


class TestToolCheckTreatmentBalance:
    """Test check_treatment_balance tool."""

    @pytest.mark.asyncio
    async def test_binary_treatment_balance(self, agent, sample_dataframe, state_with_dataframe):
        """Test checking balance for binary treatment."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_check_treatment_balance(state_with_dataframe, column="treat")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["treatment_type"] == "binary"
        assert "minority_pct" in result.output
        assert "majority_pct" in result.output
        assert "assessment" in result.output
        assert "suitable_methods" in result.output

    @pytest.mark.asyncio
    async def test_categorical_treatment(self, agent, sample_dataframe, state_with_dataframe):
        """Test checking balance for categorical variable."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_check_treatment_balance(state_with_dataframe, column="region")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_unique"] == 4

    @pytest.mark.asyncio
    async def test_continuous_variable(self, agent, sample_dataframe, state_with_dataframe):
        """Test checking balance for continuous variable."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_check_treatment_balance(state_with_dataframe, column="income")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["treatment_type"] == "continuous"
        assert result.output["assessment"] == "DOSE_RESPONSE"


class TestToolCheckRelationship:
    """Test check_column_relationship tool."""

    @pytest.mark.asyncio
    async def test_numeric_correlation(self, agent, sample_dataframe, state_with_dataframe):
        """Test checking correlation between numeric columns."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_check_relationship(
            state_with_dataframe, column1="age", column2="income"
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert "pearson_correlation" in result.output
        assert "spearman_correlation" in result.output
        assert "strength" in result.output

    @pytest.mark.asyncio
    async def test_binary_numeric_relationship(self, agent, sample_dataframe, state_with_dataframe):
        """Test checking relationship between binary and numeric."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_check_relationship(
            state_with_dataframe, column1="treat", column2="outcome"
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert "group_0_mean" in result.output
        assert "group_1_mean" in result.output
        assert "mean_difference" in result.output

    @pytest.mark.asyncio
    async def test_column_not_found(self, agent, sample_dataframe, state_with_dataframe):
        """Test error when column not found."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_check_relationship(
            state_with_dataframe, column1="nonexistent", column2="age"
        )

        assert result.status == ToolResultStatus.ERROR


class TestToolCheckTimeDimension:
    """Test check_time_dimension tool."""

    @pytest.mark.asyncio
    async def test_no_time_dimension(self, agent, sample_dataframe, state_with_dataframe):
        """Test when no time dimension exists."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_check_time_dimension(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_time_dimension"] is False

    @pytest.mark.asyncio
    async def test_with_time_column(self, agent, state_with_dataframe):
        """Test when time column exists."""
        df = pd.DataFrame({
            "year": [2019, 2020, 2021, 2022],
            "value": [1, 2, 3, 4],
        })
        agent._df = df
        agent._profile = agent._compute_basic_profile(df)

        result = await agent._tool_check_time_dimension(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_time_dimension"] is True
        assert len(result.output["candidates"]) > 0


class TestToolCheckDiscontinuity:
    """Test check_discontinuity_candidates tool."""

    @pytest.mark.asyncio
    async def test_no_discontinuity_candidates(self, agent, sample_dataframe, state_with_dataframe):
        """Test when no RDD candidates exist."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_check_discontinuity(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_rdd_candidates"] is False

    @pytest.mark.asyncio
    async def test_with_score_column(self, agent, state_with_dataframe):
        """Test when score column exists."""
        df = pd.DataFrame({
            "test_score": np.random.normal(50, 10, 100),
            "passed": np.random.binomial(1, 0.5, 100),
        })
        agent._df = df
        agent._profile = agent._compute_basic_profile(df)

        result = await agent._tool_check_discontinuity(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_rdd_candidates"] is True
        assert len(result.output["candidates"]) > 0


class TestToolFinalizeProfile:
    """Test finalize_profile tool."""

    @pytest.mark.asyncio
    async def test_finalizes_profile(self, agent, sample_dataframe, state_with_dataframe):
        """Test finalizing the profile."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = await agent._tool_finalize_profile(
            state_with_dataframe,
            treatment_candidates=["treat"],
            outcome_candidates=["outcome", "income"],
            potential_confounders=["age", "education", "gender"],
            recommended_methods=["IPW", "Matching"],
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["profile_finalized"] is True
        assert agent._finalized is True
        assert agent._final_result["treatment_candidates"] == ["treat"]
        assert agent._final_result["outcome_candidates"] == ["outcome", "income"]


class TestComputeBasicProfile:
    """Test basic profile computation."""

    def test_computes_profile(self, agent, sample_dataframe):
        """Test computing basic profile."""
        profile = agent._compute_basic_profile(sample_dataframe)

        assert profile.n_samples == 100
        assert profile.n_features == 7
        assert "treat" in profile.feature_names
        assert profile.feature_types["treat"] == "binary"
        assert profile.feature_types["income"] == "numeric"
        assert profile.feature_types["gender"] == "categorical"

    def test_handles_missing_values(self, agent):
        """Test handling missing values."""
        df = pd.DataFrame({
            "a": [1, 2, None, 4],
            "b": [None, None, 3, 4],
        })

        profile = agent._compute_basic_profile(df)

        assert profile.missing_values["a"] == 1
        assert profile.missing_values["b"] == 2


class TestAutoFinalize:
    """Test auto-finalization when agent doesn't call finalize."""

    def test_auto_finalize_finds_treatment(self, agent, sample_dataframe):
        """Test auto-finalize identifies treatment candidates."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = agent._auto_finalize()

        assert "treat" in result["treatment_candidates"]

    def test_auto_finalize_finds_outcome(self, agent, sample_dataframe):
        """Test auto-finalize identifies outcome candidates."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = agent._auto_finalize()

        # Should find numeric columns as outcomes
        assert len(result["outcome_candidates"]) > 0

    def test_auto_finalize_finds_confounders(self, agent, sample_dataframe):
        """Test auto-finalize identifies confounders."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        result = agent._auto_finalize()

        assert len(result["potential_confounders"]) > 0


class TestInitialObservation:
    """Test initial observation generation."""

    def test_generates_lean_observation(self, agent, state_with_dataframe):
        """Test that initial observation is lean, not a dump."""
        obs = agent._get_initial_observation(state_with_dataframe)

        # Should be short
        assert len(obs) < 500

        # Should mention the dataset
        assert "test_dataset" in obs or "dataset" in obs.lower()

        # Should suggest using tools
        assert "domain knowledge" in obs.lower() or "tools" in obs.lower()


class TestTaskCompletion:
    """Test task completion logic."""

    @pytest.mark.asyncio
    async def test_not_complete_initially(self, agent, state_with_dataframe):
        """Test that task is not complete initially."""
        result = await agent.is_task_complete(state_with_dataframe)
        assert result is False

    @pytest.mark.asyncio
    async def test_complete_after_finalize(self, agent, sample_dataframe, state_with_dataframe):
        """Test task is complete after finalize."""
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)

        # Finalize
        await agent._tool_finalize_profile(
            state_with_dataframe,
            treatment_candidates=["treat"],
            outcome_candidates=["outcome"],
            potential_confounders=["age"],
        )

        # Set profile on state
        state_with_dataframe.data_profile = agent._profile

        result = await agent.is_task_complete(state_with_dataframe)
        assert result is True


class TestContextToolsIntegration:
    """Test that context tools work with the profiler."""

    @pytest.mark.asyncio
    async def test_ask_domain_knowledge_available(self, agent, state_with_dataframe):
        """Test that ask_domain_knowledge tool is callable."""
        # Should not error even if domain_knowledge is None
        result = await agent._ask_domain_knowledge(
            state_with_dataframe,
            question="What is the treatment variable?"
        )

        assert result.status == ToolResultStatus.SUCCESS
        # When no domain knowledge, returns found=False
        assert result.output["found"] == False

    @pytest.mark.asyncio
    async def test_list_columns_with_profile(self, agent, sample_dataframe, state_with_dataframe):
        """Test list_columns with data profile."""
        agent._df = sample_dataframe
        profile = agent._compute_basic_profile(sample_dataframe)
        state_with_dataframe.data_profile = profile

        result = await agent._list_columns(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert "treat" in result.output["columns"]
