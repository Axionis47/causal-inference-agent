"""Unit tests for ContextTools mixin."""

import pytest

from src.agents.base import (
    AnalysisState,
    DataProfile,
    DatasetInfo,
    EDAResult,
    ReActAgent,
    ToolResult,
    ToolResultStatus,
)
from src.agents.base.context_tools import ContextTools


class MockReActAgent(ReActAgent, ContextTools):
    """Mock agent for testing ContextTools mixin."""

    AGENT_NAME = "mock_agent"
    MAX_STEPS = 5
    SYSTEM_PROMPT = "Test agent"

    def __init__(self):
        super().__init__()
        self.register_context_tools()

    def _get_initial_observation(self, state: AnalysisState) -> str:
        return "Test observation"

    async def is_task_complete(self, state: AnalysisState) -> bool:
        return True


@pytest.fixture
def mock_agent():
    """Create mock agent with context tools."""
    return MockReActAgent()


@pytest.fixture
def state_with_domain_knowledge():
    """Create state with domain knowledge."""
    return AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://kaggle.com/test", name="test"),
        domain_knowledge={
            "hypotheses": [
                {
                    "claim": "treat is the treatment variable",
                    "confidence": "high",
                    "evidence": "Description says randomly assigned"
                },
                {
                    "claim": "re78 is the outcome variable (1978 earnings)",
                    "confidence": "high",
                    "evidence": "Description explicitly states"
                },
                {
                    "claim": "age is immutable - cannot be caused by treatment",
                    "confidence": "high",
                    "evidence": "Demographic characteristic"
                },
                {
                    "claim": "re74 and re75 are potential confounders",
                    "confidence": "medium",
                    "evidence": "Pre-program earnings correlate with both treatment and outcome"
                }
            ],
            "uncertainties": [
                {
                    "issue": "Control group source unclear",
                    "impact": "May need to adjust for confounding if observational"
                }
            ],
            "temporal_understanding": "Demographics before treatment, re78 after",
            "immutable_vars": ["age", "black", "hispanic"]
        }
    )


@pytest.fixture
def state_with_profile():
    """Create state with data profile."""
    return AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://kaggle.com/test", name="test"),
        data_profile=DataProfile(
            n_samples=1000,
            n_features=10,
            feature_names=["age", "education", "treat", "re78", "re74", "re75"],
            feature_types={
                "age": "numeric",
                "education": "numeric",
                "treat": "binary",
                "re78": "numeric",
                "re74": "numeric",
                "re75": "numeric"
            },
            missing_values={"age": 0, "education": 5, "treat": 0, "re78": 0},
            numeric_stats={
                "age": {"mean": 25.0, "std": 7.0, "min": 18.0, "max": 55.0},
                "re78": {"mean": 5000.0, "std": 6000.0, "min": 0.0, "max": 60000.0}
            },
            categorical_stats={},
            treatment_candidates=["treat"],
            outcome_candidates=["re78"],
            potential_confounders=["age", "education", "re74", "re75"]
        )
    )


@pytest.fixture
def state_with_eda():
    """Create state with EDA results."""
    state = AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://kaggle.com/test", name="test"),
    )
    state.eda_result = EDAResult(
        data_quality_score=75.0,
        data_quality_issues=["Moderate missing data", "Some outliers in re78"],
        outliers={
            "re78": {"iqr_outliers": 15, "iqr_pct": 1.5},
            "age": {"iqr_outliers": 3, "iqr_pct": 0.3}
        },
        high_correlations=[
            {"var1": "re74", "var2": "re75", "correlation": 0.85},
            {"var1": "age", "var2": "education", "correlation": 0.72}
        ],
        covariate_balance={
            "age": {"smd": 0.05, "is_balanced": True},
            "education": {"smd": 0.15, "is_balanced": False},
            "re74": {"smd": 0.25, "is_balanced": False}
        },
        balance_summary="2 of 3 covariates imbalanced",
        vif_scores={"age": 2.1, "education": 3.5, "re74": 8.2},
        multicollinearity_warnings=["Moderate VIF for re74"],
        distribution_stats={
            "re78": {"skewness": 1.5, "kurtosis": 2.0},
            "age": {"skewness": 0.2, "kurtosis": 0.1}
        }
    )
    return state


class TestAskDomainKnowledge:
    """Test ask_domain_knowledge tool."""

    @pytest.mark.asyncio
    async def test_finds_treatment_hypothesis(self, mock_agent, state_with_domain_knowledge):
        """Test finding treatment-related hypothesis."""
        result = await mock_agent._ask_domain_knowledge(
            state_with_domain_knowledge,
            question="What is the treatment variable?"
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["found"] is True
        findings = result.output["findings"]
        assert any("treat" in str(f).lower() and "treatment" in str(f).lower() for f in findings)

    @pytest.mark.asyncio
    async def test_finds_outcome_hypothesis(self, mock_agent, state_with_domain_knowledge):
        """Test finding outcome-related hypothesis."""
        result = await mock_agent._ask_domain_knowledge(
            state_with_domain_knowledge,
            question="What is the outcome variable?"
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["found"] is True

    @pytest.mark.asyncio
    async def test_finds_immutable_vars(self, mock_agent, state_with_domain_knowledge):
        """Test finding immutable variables."""
        result = await mock_agent._ask_domain_knowledge(
            state_with_domain_knowledge,
            question="Which variables are immutable?"
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["found"] is True
        # Should find the immutable_vars entry
        findings = result.output["findings"]
        assert any("immutable" in str(f).lower() for f in findings)

    @pytest.mark.asyncio
    async def test_finds_temporal_ordering(self, mock_agent, state_with_domain_knowledge):
        """Test finding temporal ordering."""
        result = await mock_agent._ask_domain_knowledge(
            state_with_domain_knowledge,
            question="What is the temporal ordering?"
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["found"] is True

    @pytest.mark.asyncio
    async def test_finds_confounders(self, mock_agent, state_with_domain_knowledge):
        """Test finding confounder hypotheses."""
        result = await mock_agent._ask_domain_knowledge(
            state_with_domain_knowledge,
            question="What are the potential confounders?"
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["found"] is True

    @pytest.mark.asyncio
    async def test_returns_uncertainties(self, mock_agent, state_with_domain_knowledge):
        """Test that uncertainties are searchable."""
        result = await mock_agent._ask_domain_knowledge(
            state_with_domain_knowledge,
            question="What are you uncertain about regarding the control group?"
        )

        assert result.status == ToolResultStatus.SUCCESS
        # Should find the control group uncertainty
        if result.output["found"]:
            findings = result.output["findings"]
            assert any("control" in str(f).lower() or "uncertainty" in str(f).lower() for f in findings)

    @pytest.mark.asyncio
    async def test_handles_no_domain_knowledge(self, mock_agent):
        """Test handling when no domain knowledge exists."""
        state = AnalysisState(
            job_id="test",
            dataset_info=DatasetInfo(url="test", name="test"),
            domain_knowledge=None
        )

        result = await mock_agent._ask_domain_knowledge(state, question="What is the treatment?")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["found"] is False
        assert "no domain knowledge" in result.output["message"].lower()

    @pytest.mark.asyncio
    async def test_handles_irrelevant_question(self, mock_agent, state_with_domain_knowledge):
        """Test handling question with no relevant findings."""
        result = await mock_agent._ask_domain_knowledge(
            state_with_domain_knowledge,
            question="What is the weather like?"
        )

        assert result.status == ToolResultStatus.SUCCESS
        # Should return not found for irrelevant questions
        assert result.output["found"] is False or len(result.output.get("findings", [])) == 0


class TestGetColumnInfo:
    """Test get_column_info tool."""

    @pytest.mark.asyncio
    async def test_gets_column_info(self, mock_agent, state_with_profile):
        """Test getting column information."""
        result = await mock_agent._get_column_info(state_with_profile, column="age")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["name"] == "age"
        assert result.output["dtype"] == "numeric"
        assert "stats" in result.output

    @pytest.mark.asyncio
    async def test_identifies_treatment_candidate(self, mock_agent, state_with_profile):
        """Test that treatment candidates are flagged."""
        result = await mock_agent._get_column_info(state_with_profile, column="treat")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output.get("is_treatment_candidate") is True

    @pytest.mark.asyncio
    async def test_identifies_outcome_candidate(self, mock_agent, state_with_profile):
        """Test that outcome candidates are flagged."""
        result = await mock_agent._get_column_info(state_with_profile, column="re78")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output.get("is_outcome_candidate") is True

    @pytest.mark.asyncio
    async def test_handles_missing_column(self, mock_agent, state_with_profile):
        """Test handling non-existent column."""
        result = await mock_agent._get_column_info(state_with_profile, column="nonexistent")

        assert result.status == ToolResultStatus.ERROR
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handles_no_profile(self, mock_agent):
        """Test handling when no profile exists."""
        state = AnalysisState(
            job_id="test",
            dataset_info=DatasetInfo(url="test", name="test")
        )

        result = await mock_agent._get_column_info(state, column="age")

        assert result.status == ToolResultStatus.ERROR
        assert "not available" in result.error.lower()


class TestGetDatasetSummary:
    """Test get_dataset_summary tool."""

    @pytest.mark.asyncio
    async def test_gets_summary_with_profile(self, mock_agent, state_with_profile):
        """Test getting dataset summary."""
        result = await mock_agent._get_dataset_summary(state_with_profile)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["profiled"] is True
        assert result.output["n_samples"] == 1000
        assert result.output["n_features"] == 10
        assert "treatment_candidates" in result.output

    @pytest.mark.asyncio
    async def test_gets_summary_without_profile(self, mock_agent):
        """Test getting summary without full profile."""
        state = AnalysisState(
            job_id="test",
            dataset_info=DatasetInfo(
                url="test",
                name="test_data",
                n_samples=500,
                n_features=8
            )
        )

        result = await mock_agent._get_dataset_summary(state)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["profiled"] is False
        assert result.output["name"] == "test_data"


class TestGetEdaFinding:
    """Test get_eda_finding tool."""

    @pytest.mark.asyncio
    async def test_gets_outlier_findings(self, mock_agent, state_with_eda):
        """Test getting outlier findings."""
        result = await mock_agent._get_eda_finding(state_with_eda, topic="outliers")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_outliers"] is True
        assert result.output["columns_with_outliers"] == 2

    @pytest.mark.asyncio
    async def test_gets_correlation_findings(self, mock_agent, state_with_eda):
        """Test getting correlation findings."""
        result = await mock_agent._get_eda_finding(state_with_eda, topic="correlations")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_high_correlations"] is True
        assert result.output["count"] == 2

    @pytest.mark.asyncio
    async def test_gets_balance_findings(self, mock_agent, state_with_eda):
        """Test getting covariate balance findings."""
        result = await mock_agent._get_eda_finding(state_with_eda, topic="balance")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["checked"] is True
        assert result.output["imbalanced_count"] == 2

    @pytest.mark.asyncio
    async def test_gets_multicollinearity_findings(self, mock_agent, state_with_eda):
        """Test getting multicollinearity findings."""
        result = await mock_agent._get_eda_finding(state_with_eda, topic="multicollinearity")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["checked"] is True
        assert len(result.output["high_vif_columns"]) > 0

    @pytest.mark.asyncio
    async def test_gets_quality_score(self, mock_agent, state_with_eda):
        """Test getting quality score."""
        result = await mock_agent._get_eda_finding(state_with_eda, topic="quality_score")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["quality_score"] == 75.0

    @pytest.mark.asyncio
    async def test_handles_no_eda(self, mock_agent):
        """Test handling when no EDA results exist."""
        state = AnalysisState(
            job_id="test",
            dataset_info=DatasetInfo(url="test", name="test")
        )

        result = await mock_agent._get_eda_finding(state, topic="outliers")

        assert result.status == ToolResultStatus.ERROR
        assert "not available" in result.error.lower()


class TestGetPreviousFinding:
    """Test get_previous_finding tool."""

    @pytest.mark.asyncio
    async def test_gets_domain_knowledge_findings(self, mock_agent, state_with_domain_knowledge):
        """Test getting domain knowledge findings."""
        result = await mock_agent._get_previous_finding(
            state_with_domain_knowledge,
            agent="domain_knowledge"
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["available"] is True
        assert result.output["hypotheses_count"] == 4

    @pytest.mark.asyncio
    async def test_gets_profiler_findings(self, mock_agent, state_with_profile):
        """Test getting profiler findings."""
        result = await mock_agent._get_previous_finding(state_with_profile, agent="data_profiler")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["available"] is True
        assert result.output["treatment_candidates"] == ["treat"]

    @pytest.mark.asyncio
    async def test_gets_eda_findings(self, mock_agent, state_with_eda):
        """Test getting EDA findings."""
        result = await mock_agent._get_previous_finding(state_with_eda, agent="eda_agent")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["available"] is True
        assert result.output["quality_score"] == 75.0

    @pytest.mark.asyncio
    async def test_handles_unavailable_agent(self, mock_agent):
        """Test handling when agent hasn't run."""
        state = AnalysisState(
            job_id="test",
            dataset_info=DatasetInfo(url="test", name="test")
        )

        result = await mock_agent._get_previous_finding(state, agent="domain_knowledge")

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["available"] is False


class TestGetTreatmentOutcome:
    """Test get_treatment_outcome tool."""

    @pytest.mark.asyncio
    async def test_gets_specified_treatment_outcome(self, mock_agent):
        """Test getting user-specified treatment/outcome."""
        state = AnalysisState(
            job_id="test",
            dataset_info=DatasetInfo(url="test", name="test"),
            treatment_variable="treat",
            outcome_variable="re78"
        )

        result = await mock_agent._get_treatment_outcome(state)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["treatment"] == "treat"
        assert result.output["outcome"] == "re78"
        assert result.output["confirmed"] is True

    @pytest.mark.asyncio
    async def test_handles_unspecified(self, mock_agent):
        """Test handling when treatment/outcome not specified."""
        state = AnalysisState(
            job_id="test",
            dataset_info=DatasetInfo(url="test", name="test")
        )

        result = await mock_agent._get_treatment_outcome(state)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["confirmed"] is False


class TestContextToolsRegistration:
    """Test that context tools are properly registered."""

    def test_all_tools_registered(self, mock_agent):
        """Test that all expected tools are registered."""
        tool_names = list(mock_agent._tools.keys())

        expected_tools = [
            "ask_domain_knowledge",
            "get_column_info",
            "get_dataset_summary",
            "list_columns",
            "get_eda_finding",
            "get_previous_finding",
            "get_treatment_outcome",
            "finish",  # Built-in ReAct tool
            "reflect",  # Built-in ReAct tool
        ]

        for tool in expected_tools:
            assert tool in tool_names, f"Tool '{tool}' not registered"

    def test_tool_schemas_exist(self, mock_agent):
        """Test that tool schemas are properly defined."""
        assert len(mock_agent._tool_schemas) > 0

        for schema in mock_agent._tool_schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
