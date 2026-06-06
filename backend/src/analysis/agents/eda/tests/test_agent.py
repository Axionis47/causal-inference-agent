"""Agent-level tests: initialization, completion, initial observation, context-tools."""

import pytest

from src.analysis.agents.base import EDAResult, ToolResultStatus


class TestInitialization:
    def test_agent_name(self, agent):
        assert agent.AGENT_NAME == "eda_agent"

    def test_max_steps(self, agent):
        assert agent.MAX_STEPS == 15

    def test_tools_registered(self, agent):
        """EDA tools, context-tools mixin, and ReAct built-ins all in the registry."""
        tool_names = list(agent._tools.keys())
        expected = [
            "get_data_overview",
            "analyze_variable",
            "detect_outliers",
            "compute_correlations",
            "compute_vif",
            "check_covariate_balance",
            "check_missing_patterns",
            "finalize_eda",
            "ask_domain_knowledge",
            "get_eda_finding",
            "get_previous_finding",
            "get_treatment_outcome",
            "get_confounder_analysis",
            "analyze_variable_semantics",
            "get_dag_adjustment_set",
            "finish",
            "reflect",
        ]
        for tool in expected:
            assert tool in tool_names, f"Tool '{tool}' not registered"

    def test_has_context_tools(self, agent):
        assert "ask_domain_knowledge" in agent._tools
        assert "get_dag_adjustment_set" in agent._tools


class TestInitialObservation:
    def test_generates_lean_observation(self, agent, state_with_dataframe):
        """Initial observation names the dataset and the workflow, not a data dump."""
        obs = agent._get_initial_observation(state_with_dataframe)
        assert len(obs) < 600
        assert "test_dataset" in obs or "dataset" in obs.lower()
        assert "domain knowledge" in obs.lower() or "treatment" in obs.lower()


class TestTaskCompletion:
    @pytest.mark.asyncio
    async def test_not_complete_initially(self, agent, state_with_dataframe):
        assert await agent.is_task_complete(state_with_dataframe) is False

    @pytest.mark.asyncio
    async def test_complete_after_finalize(self, agent, sample_dataframe, state_with_dataframe):
        """Completion requires both the _finalized flag AND eda_result on state."""
        agent._df = sample_dataframe
        agent._eda_result = EDAResult()

        await agent._tool_finalize(
            state_with_dataframe,
            data_quality_score=85.0,
            key_findings=["Finding"],
            data_quality_issues=[],
            recommendations=["Recommendation"],
            causal_readiness="ready",
        )
        state_with_dataframe.eda_result = agent._eda_result

        assert await agent.is_task_complete(state_with_dataframe) is True


class TestContextToolsIntegration:
    @pytest.mark.asyncio
    async def test_ask_domain_knowledge_available(self, agent, state_with_dataframe):
        """ask_domain_knowledge returns found=False cleanly when no DK is on state."""
        result = await agent._ask_domain_knowledge(
            state_with_dataframe,
            question="What is the treatment variable?",
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["found"] is False

    @pytest.mark.asyncio
    async def test_list_columns_with_profile(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        result = await agent._list_columns(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert "treat" in result.output["columns"]
