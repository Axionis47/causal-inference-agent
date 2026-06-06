"""Agent-level tests: initialization, completion, initial observation, context-tools."""

import pytest

from src.analysis.agents.base import CausalDAG, ToolResultStatus


class TestInitialization:
    def test_agent_name(self, agent):
        assert agent.AGENT_NAME == "causal_discovery"

    def test_max_steps(self, agent):
        assert agent.MAX_STEPS == 15

    def test_tools_registered(self, agent):
        """Discovery tools, context-tools mixin, and ReAct built-ins all in the registry."""
        tool_names = list(agent._tools.keys())
        expected = [
            "get_data_characteristics",
            "run_discovery_algorithm",
            "inspect_graph",
            "validate_graph",
            "compare_algorithms",
            "finalize_discovery",
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
        assert "Treatment" in obs or "treatment" in obs
        assert "Outcome" in obs or "outcome" in obs


class TestTaskCompletion:
    @pytest.mark.asyncio
    async def test_not_complete_initially(self, agent, state_with_dataframe):
        assert await agent.is_task_complete(state_with_dataframe) is False

    @pytest.mark.asyncio
    async def test_complete_after_finalize(self, agent, state_with_dataframe):
        """Completion requires both _finalized AND a discovered_dag on state."""
        await agent._tool_finalize_discovery(
            state_with_dataframe,
            chosen_algorithm="pc",
            interpretation="Test",
            confidence="high",
        )
        state_with_dataframe.discovered_dag = CausalDAG(
            nodes=["a", "b"],
            edges=[],
            discovery_method="test",
        )
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
    async def test_list_columns_with_profile(self, agent, state_with_dataframe):
        result = await agent._list_columns(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert "treat" in result.output["columns"]
