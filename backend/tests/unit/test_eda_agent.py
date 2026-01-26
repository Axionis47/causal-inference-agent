"""Unit tests for EDAAgent (ReAct-based)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock

from src.agents.base import AnalysisState, DatasetInfo, DataProfile, ToolResultStatus
from src.agents.specialists.eda_agent import EDAAgent


@pytest.fixture
def agent():
    """Create agent instance."""
    return EDAAgent()


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        "treat": np.random.binomial(1, 0.4, n),  # Binary treatment (40% treated)
        "age": np.random.normal(40, 10, n),
        "income": np.random.normal(50000, 15000, n),
        "education": np.random.randint(10, 20, n),
        "outcome": np.random.normal(100, 20, n),
        "gender": np.random.choice(["M", "F"], n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
    })


@pytest.fixture
def sample_profile(sample_dataframe):
    """Create a sample data profile."""
    profile = DataProfile(
        n_samples=100,
        n_features=7,
        feature_names=list(sample_dataframe.columns),
        feature_types={
            "treat": "binary",
            "age": "numeric",
            "income": "numeric",
            "education": "numeric",
            "outcome": "numeric",
            "gender": "categorical",
            "region": "categorical",
        },
        missing_values={col: 0 for col in sample_dataframe.columns},
        numeric_stats={
            "treat": {"mean": 0.4, "std": 0.49, "min": 0.0, "max": 1.0},
            "age": {"mean": 40.0, "std": 10.0, "min": 20.0, "max": 60.0},
            "income": {"mean": 50000.0, "std": 15000.0, "min": 20000.0, "max": 80000.0},
            "education": {"mean": 15.0, "std": 3.0, "min": 10.0, "max": 20.0},
            "outcome": {"mean": 100.0, "std": 20.0, "min": 60.0, "max": 140.0},
        },
        categorical_stats={
            "gender": {"M": 50, "F": 50},
            "region": {"North": 25, "South": 25, "East": 25, "West": 25},
        },
        treatment_candidates=["treat"],
        outcome_candidates=["outcome", "income"],
        potential_confounders=["age", "education"],
    )
    return profile


@pytest.fixture
def state_with_dataframe(sample_dataframe, sample_profile, tmp_path):
    """Create state with a loaded dataframe."""
    import pickle

    # Save dataframe to temp path
    pkl_path = tmp_path / "test_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(sample_dataframe, f)

    return AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(
            url="https://kaggle.com/test/dataset",
            name="test_dataset",
            local_path=str(pkl_path),
        ),
        dataframe_path=str(pkl_path),
        data_profile=sample_profile,
    )


class TestInitialization:
    """Test agent initialization."""

    def test_agent_name(self, agent):
        """Test agent has correct name."""
        assert agent.AGENT_NAME == "eda_agent"

    def test_max_steps(self, agent):
        """Test agent has reasonable max steps."""
        assert agent.MAX_STEPS == 15

    def test_tools_registered(self, agent):
        """Test that EDA tools are registered."""
        tool_names = list(agent._tools.keys())

        expected_tools = [
            # EDA tools
            "get_data_overview",
            "analyze_variable",
            "detect_outliers",
            "compute_correlations",
            "compute_vif",
            "check_covariate_balance",
            "check_missing_patterns",
            "finalize_eda",
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
    """Test get_data_overview tool."""

    @pytest.mark.asyncio
    async def test_returns_overview(self, agent, sample_dataframe, state_with_dataframe):
        """Test getting dataset overview."""
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"

        result = await agent._tool_get_overview(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_samples"] == 100
        assert result.output["n_features"] == 7
        assert "numeric_columns" in result.output
        assert "categorical_columns" in result.output
        assert "binary_columns" in result.output

    @pytest.mark.asyncio
    async def test_includes_treatment_info(self, agent, sample_dataframe, state_with_dataframe):
        """Test that treatment info is included."""
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"

        result = await agent._tool_get_overview(state_with_dataframe)

        assert result.output["treatment"] is not None
        assert result.output["treatment"]["variable"] == "treat"

    @pytest.mark.asyncio
    async def test_includes_outcome_info(self, agent, sample_dataframe, state_with_dataframe):
        """Test that outcome info is included."""
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"

        result = await agent._tool_get_overview(state_with_dataframe)

        assert result.output["outcome"] is not None
        assert result.output["outcome"]["variable"] == "outcome"

    @pytest.mark.asyncio
    async def test_error_when_no_dataframe(self, agent, state_with_dataframe):
        """Test error when dataframe not loaded."""
        agent._df = None

        result = await agent._tool_get_overview(state_with_dataframe)

        assert result.status == ToolResultStatus.ERROR
        assert "not loaded" in result.error


class TestToolAnalyzeVariable:
    """Test analyze_variable tool."""

    @pytest.mark.asyncio
    async def test_analyzes_numeric_variable(self, agent, sample_dataframe, state_with_dataframe):
        """Test analyzing a numeric variable."""
        agent._df = sample_dataframe

        result = await agent._tool_analyze_variable(state_with_dataframe, variable="income")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["variable"] == "income"
        assert result.output["type"] == "numeric"
        assert "mean" in result.output
        assert "std" in result.output
        assert "skewness" in result.output

    @pytest.mark.asyncio
    async def test_includes_normality_tests(self, agent, sample_dataframe, state_with_dataframe):
        """Test that normality tests are included."""
        agent._df = sample_dataframe

        result = await agent._tool_analyze_variable(
            state_with_dataframe, variable="income", include_normality_tests=True
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert "normality_tests" in result.output

    @pytest.mark.asyncio
    async def test_analyzes_categorical_variable(self, agent, sample_dataframe, state_with_dataframe):
        """Test analyzing a categorical variable."""
        agent._df = sample_dataframe

        result = await agent._tool_analyze_variable(state_with_dataframe, variable="gender")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["type"] == "categorical"
        assert "top_values" in result.output

    @pytest.mark.asyncio
    async def test_variable_not_found(self, agent, sample_dataframe, state_with_dataframe):
        """Test error when variable not found."""
        agent._df = sample_dataframe

        result = await agent._tool_analyze_variable(state_with_dataframe, variable="nonexistent")

        assert result.status == ToolResultStatus.ERROR
        assert "not found" in result.error


class TestToolDetectOutliers:
    """Test detect_outliers tool."""

    @pytest.mark.asyncio
    async def test_detects_outliers(self, agent, sample_dataframe, state_with_dataframe):
        """Test outlier detection."""
        agent._df = sample_dataframe
        agent._treatment_var = "treat"

        result = await agent._tool_detect_outliers(state_with_dataframe, method="both")

        assert result.status == ToolResultStatus.SUCCESS
        assert "variables_checked" in result.output
        assert "variables_with_outliers" in result.output

    @pytest.mark.asyncio
    async def test_iqr_method(self, agent, sample_dataframe, state_with_dataframe):
        """Test IQR method specifically."""
        agent._df = sample_dataframe
        agent._treatment_var = "treat"

        result = await agent._tool_detect_outliers(state_with_dataframe, method="iqr")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["method"] == "iqr"

    @pytest.mark.asyncio
    async def test_zscore_method(self, agent, sample_dataframe, state_with_dataframe):
        """Test Z-score method specifically."""
        agent._df = sample_dataframe
        agent._treatment_var = "treat"

        result = await agent._tool_detect_outliers(state_with_dataframe, method="zscore")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["method"] == "zscore"


class TestToolComputeCorrelations:
    """Test compute_correlations tool."""

    @pytest.mark.asyncio
    async def test_computes_correlations(self, agent, sample_dataframe, state_with_dataframe):
        """Test correlation computation."""
        from src.agents.base import EDAResult
        agent._df = sample_dataframe
        agent._eda_result = EDAResult()

        result = await agent._tool_compute_correlations(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert "n_variables" in result.output
        assert "high_correlations_count" in result.output

    @pytest.mark.asyncio
    async def test_spearman_method(self, agent, sample_dataframe, state_with_dataframe):
        """Test Spearman correlation method."""
        from src.agents.base import EDAResult
        agent._df = sample_dataframe
        agent._eda_result = EDAResult()

        result = await agent._tool_compute_correlations(state_with_dataframe, method="spearman")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["method"] == "spearman"

    @pytest.mark.asyncio
    async def test_custom_threshold(self, agent, sample_dataframe, state_with_dataframe):
        """Test custom correlation threshold."""
        from src.agents.base import EDAResult
        agent._df = sample_dataframe
        agent._eda_result = EDAResult()

        result = await agent._tool_compute_correlations(state_with_dataframe, threshold=0.5)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["threshold"] == 0.5


class TestToolComputeVIF:
    """Test compute_vif tool."""

    @pytest.mark.asyncio
    async def test_computes_vif(self, agent, sample_dataframe, state_with_dataframe):
        """Test VIF computation."""
        from src.agents.base import EDAResult
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        agent._eda_result = EDAResult()

        result = await agent._tool_compute_vif(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert "n_covariates" in result.output
        assert "top_vif" in result.output

    @pytest.mark.asyncio
    async def test_vif_with_specific_covariates(self, agent, sample_dataframe, state_with_dataframe):
        """Test VIF with specific covariates."""
        from src.agents.base import EDAResult
        agent._df = sample_dataframe
        agent._eda_result = EDAResult()

        result = await agent._tool_compute_vif(
            state_with_dataframe, covariates=["age", "income", "education"]
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_covariates"] == 3


class TestToolCheckBalance:
    """Test check_covariate_balance tool."""

    @pytest.mark.asyncio
    async def test_checks_balance(self, agent, sample_dataframe, state_with_dataframe):
        """Test covariate balance check."""
        from src.agents.base import EDAResult
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        agent._eda_result = EDAResult()

        result = await agent._tool_check_balance(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert "treatment_variable" in result.output
        assert "n_treated" in result.output
        assert "n_control" in result.output
        assert "n_imbalanced" in result.output

    @pytest.mark.asyncio
    async def test_balance_with_specific_covariates(self, agent, sample_dataframe, state_with_dataframe):
        """Test balance with specific covariates."""
        from src.agents.base import EDAResult
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._eda_result = EDAResult()

        result = await agent._tool_check_balance(
            state_with_dataframe, covariates=["age", "income"]
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_covariates_checked"] == 2

    @pytest.mark.asyncio
    async def test_balance_no_treatment(self, agent, sample_dataframe, state_with_dataframe):
        """Test balance check without treatment variable."""
        from src.agents.base import EDAResult
        agent._df = sample_dataframe
        agent._treatment_var = None
        agent._eda_result = EDAResult()

        result = await agent._tool_check_balance(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert "error" in result.output


class TestToolCheckMissing:
    """Test check_missing_patterns tool."""

    @pytest.mark.asyncio
    async def test_no_missing(self, agent, sample_dataframe, state_with_dataframe):
        """Test when no missing values."""
        agent._df = sample_dataframe

        result = await agent._tool_check_missing(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_missing"] == False

    @pytest.mark.asyncio
    async def test_with_missing(self, agent, state_with_dataframe):
        """Test when there are missing values."""
        df = pd.DataFrame({
            "a": [1, 2, None, 4, 5],
            "b": [None, None, 3, 4, 5],
            "treat": [0, 1, 0, 1, 0],
        })
        agent._df = df
        agent._treatment_var = "treat"

        result = await agent._tool_check_missing(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_missing"] == True
        assert result.output["n_cols_with_missing"] == 2


class TestToolFinalize:
    """Test finalize_eda tool."""

    @pytest.mark.asyncio
    async def test_finalizes_eda(self, agent, sample_dataframe, state_with_dataframe):
        """Test finalizing the EDA."""
        from src.agents.base import EDAResult
        agent._df = sample_dataframe
        agent._eda_result = EDAResult()

        result = await agent._tool_finalize(
            state_with_dataframe,
            data_quality_score=85.0,
            key_findings=["Good data quality", "Some outliers"],
            data_quality_issues=["Minor outliers in income"],
            recommendations=["Use robust methods"],
            causal_readiness="ready",
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["eda_finalized"] == True
        assert agent._finalized == True
        assert agent._final_result["data_quality_score"] == 85.0
        assert agent._final_result["causal_readiness"] == "ready"


class TestAutoFinalize:
    """Test auto-finalization when agent doesn't call finalize."""

    def test_auto_finalize_no_issues(self, agent, sample_dataframe):
        """Test auto-finalize with no issues."""
        agent._df = sample_dataframe
        agent._missing_analysis = {"has_missing": False}
        agent._outlier_results = {}
        agent._vif_results = {}
        agent._balance_results = {}

        result = agent._auto_finalize()

        assert result["data_quality_score"] >= 80
        assert result["causal_readiness"] == "ready"

    def test_auto_finalize_with_missing_data(self, agent, sample_dataframe):
        """Test auto-finalize with missing data."""
        agent._df = sample_dataframe
        agent._missing_analysis = {
            "has_missing": True,
            "total_missing_pct": 15.0,
        }
        agent._outlier_results = {}
        agent._vif_results = {}
        agent._balance_results = {}

        result = agent._auto_finalize()

        assert result["data_quality_score"] < 100
        assert "missing data" in " ".join(result["data_quality_issues"]).lower()

    def test_auto_finalize_with_outliers(self, agent, sample_dataframe):
        """Test auto-finalize with outliers."""
        agent._df = sample_dataframe
        agent._missing_analysis = {"has_missing": False}
        agent._outlier_results = {
            "col1": {"iqr_outliers": 5},
            "col2": {"iqr_outliers": 3},
            "col3": {"iqr_outliers": 4},
        }
        agent._vif_results = {}
        agent._balance_results = {}

        result = agent._auto_finalize()

        assert result["data_quality_score"] < 100
        assert "outlier" in " ".join(result["recommendations"]).lower()

    def test_auto_finalize_with_multicollinearity(self, agent, sample_dataframe):
        """Test auto-finalize with multicollinearity."""
        agent._df = sample_dataframe
        agent._missing_analysis = {"has_missing": False}
        agent._outlier_results = {}
        agent._vif_results = {"var1": 12.0, "var2": 8.0}  # High VIF
        agent._balance_results = {}

        result = agent._auto_finalize()

        assert result["data_quality_score"] < 100
        assert "multicollinearity" in " ".join(result["data_quality_issues"]).lower()

    def test_auto_finalize_with_imbalance(self, agent, sample_dataframe):
        """Test auto-finalize with covariate imbalance."""
        agent._df = sample_dataframe
        agent._missing_analysis = {"has_missing": False}
        agent._outlier_results = {}
        agent._vif_results = {}
        agent._balance_results = {
            "age": {"is_balanced": False, "smd": 0.3},
            "income": {"is_balanced": False, "smd": 0.25},
            "education": {"is_balanced": True, "smd": 0.05},
        }

        result = agent._auto_finalize()

        assert result["data_quality_score"] < 100
        assert "imbalanced" in " ".join(result["data_quality_issues"]).lower()


class TestInitialObservation:
    """Test initial observation generation."""

    def test_generates_lean_observation(self, agent, state_with_dataframe):
        """Test that initial observation is lean, not a dump."""
        obs = agent._get_initial_observation(state_with_dataframe)

        # Should be short
        assert len(obs) < 600

        # Should mention the dataset
        assert "test_dataset" in obs or "dataset" in obs.lower()

        # Should suggest workflow
        assert "domain knowledge" in obs.lower() or "treatment" in obs.lower()


class TestTaskCompletion:
    """Test task completion logic."""

    @pytest.mark.asyncio
    async def test_not_complete_initially(self, agent, state_with_dataframe):
        """Test that task is not complete initially."""
        result = await agent.is_task_complete(state_with_dataframe)
        assert result == False

    @pytest.mark.asyncio
    async def test_complete_after_finalize(self, agent, sample_dataframe, state_with_dataframe):
        """Test task is complete after finalize."""
        from src.agents.base import EDAResult
        agent._df = sample_dataframe
        agent._eda_result = EDAResult()

        # Finalize
        await agent._tool_finalize(
            state_with_dataframe,
            data_quality_score=85.0,
            key_findings=["Finding"],
            data_quality_issues=[],
            recommendations=["Recommendation"],
            causal_readiness="ready",
        )

        # Set EDA result on state
        state_with_dataframe.eda_result = agent._eda_result

        result = await agent.is_task_complete(state_with_dataframe)
        assert result == True


class TestContextToolsIntegration:
    """Test that context tools work with the EDA agent."""

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

        result = await agent._list_columns(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert "treat" in result.output["columns"]


class TestPopulateEDAResult:
    """Test EDA result population."""

    def test_populates_from_final_result(self, agent, sample_dataframe):
        """Test that EDA result is populated from final result."""
        from src.agents.base import EDAResult
        agent._eda_result = EDAResult()
        agent._analyzed_distributions = {"age": {"mean": 40}}
        agent._outlier_results = {"income": {"iqr_outliers": 5}}
        agent._vif_results = {"age": 1.5}
        agent._balance_results = {"age": {"smd": 0.05, "is_balanced": True}}

        final_result = {
            "data_quality_score": 75.0,
            "key_findings": ["Finding 1"],
            "data_quality_issues": ["Issue 1"],
            "recommendations": ["Rec 1"],
            "causal_readiness": "needs_attention",
        }

        agent._populate_eda_result(final_result)

        assert agent._eda_result.data_quality_score == 75.0
        assert agent._eda_result.data_quality_issues == ["Issue 1"]
        assert agent._eda_result.distribution_stats == {"age": {"mean": 40}}
        assert agent._eda_result.outliers == {"income": {"iqr_outliers": 5}}
