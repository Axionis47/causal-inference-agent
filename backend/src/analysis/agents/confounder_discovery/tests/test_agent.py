"""Agent-level tests: initialization and completion."""

import pytest


class TestInitialization:
    def test_agent_name(self, agent):
        assert agent.AGENT_NAME == "confounder_discovery"

    def test_max_steps(self, agent):
        assert agent.MAX_STEPS == 15

    def test_tools_registered(self, agent):
        """All 5 discovery tools, the context-tools mixin, and ReAct built-ins are present."""
        tool_names = list(agent._tools.keys())
        expected = [
            "get_candidate_variables",
            "compute_correlation",
            "compute_partial_correlation",
            "test_confounder_criteria",
            "finalize_confounders",
            "ask_domain_knowledge",
            "get_eda_finding",
            "finish",
            "reflect",
        ]
        for tool in expected:
            assert tool in tool_names, f"Tool '{tool}' not registered"


class TestTaskCompletion:
    @pytest.mark.asyncio
    async def test_not_complete_initially(self, agent, state):
        assert await agent.is_task_complete(state) is False

    @pytest.mark.asyncio
    async def test_complete_after_flag_flip(self, agent, state):
        """Completion is driven purely by the _finalized flag."""
        agent._finalized = True
        assert await agent.is_task_complete(state) is True
