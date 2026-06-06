"""Agent-level tests: initialization and completion."""

import pytest


class TestInitialization:
    def test_agent_name(self, agent):
        assert agent.AGENT_NAME == "data_repair"

    def test_max_steps(self, agent):
        assert agent.MAX_STEPS == 20

    def test_repair_tools_registered(self, agent):
        """All 10 repair tools, the context-tools mixin, and ReAct built-ins are present."""
        tool_names = list(agent._tools.keys())
        expected = [
            "get_data_summary",
            "check_missing_values",
            "check_outliers",
            "check_collinearity",
            "check_skewness",
            "repair_missing",
            "repair_outliers",
            "repair_collinearity",
            "validate_current_state",
            "finalize_repairs",
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
    async def test_complete_after_finalize_flag(self, agent, state):
        """Completion is driven purely by the agent's _finalized flag."""
        agent._finalized = True
        assert await agent.is_task_complete(state) is True
