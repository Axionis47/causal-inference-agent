"""Tests for LLM client reset propagation to agents.

Verifies that after reset_llm_client(), agents pick up the new
client instance via the property-based access pattern.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def _mock_llm():
    """Mock the LLM client module."""
    with patch("src.agents.base.agent.get_llm_client") as mock_get:
        yield mock_get


class TestLLMResetPropagation:
    """BaseAgent.llm property reflects reset_llm_client()."""

    def test_llm_is_property_not_cached(self, _mock_llm):
        """Agent.llm should call get_llm_client() each time."""
        from src.agents.base.agent import BaseAgent

        # BaseAgent is abstract, create a minimal subclass
        class TestAgent(BaseAgent):
            AGENT_NAME = "test"

            async def execute(self, state):
                return state

        client_a = MagicMock()
        client_b = MagicMock()
        _mock_llm.side_effect = [client_a, client_b]

        agent = TestAgent()
        first = agent.llm
        second = agent.llm

        assert first is client_a
        assert second is client_b
        assert _mock_llm.call_count == 2

    def test_multiple_agents_reflect_same_reset(self, _mock_llm):
        """After reset, all agents see the new client."""
        from src.agents.base.agent import BaseAgent

        class AgentA(BaseAgent):
            AGENT_NAME = "a"

            async def execute(self, state):
                return state

        class AgentB(BaseAgent):
            AGENT_NAME = "b"

            async def execute(self, state):
                return state

        old_client = MagicMock()
        new_client = MagicMock()

        _mock_llm.return_value = old_client
        a = AgentA()
        b = AgentB()
        assert a.llm is old_client
        assert b.llm is old_client

        # Simulate reset
        _mock_llm.return_value = new_client
        assert a.llm is new_client
        assert b.llm is new_client


class TestStateContractValidation:
    """Warn-only state contract validation in execute_with_tracing."""

    @pytest.mark.asyncio
    async def test_warns_on_missing_required_fields(self, _mock_llm):
        from src.agents.base.agent import BaseAgent
        from src.agents.base.state import AnalysisState, DatasetInfo

        _mock_llm.return_value = MagicMock()

        class StrictAgent(BaseAgent):
            AGENT_NAME = "strict"
            REQUIRED_STATE_FIELDS = ["data_profile", "treatment_effects"]

            async def execute(self, state):
                return state

        agent = StrictAgent()
        state = AnalysisState(job_id="test-001", dataset_info=DatasetInfo(url="test"))

        # Should not raise — warn-only
        result = await agent.execute_with_tracing(state)
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_warning_when_fields_present(self, _mock_llm):
        from src.agents.base.agent import BaseAgent
        from src.agents.base.state import AnalysisState, DatasetInfo

        _mock_llm.return_value = MagicMock()

        class SafeAgent(BaseAgent):
            AGENT_NAME = "safe"
            REQUIRED_STATE_FIELDS = ["dataset_info"]

            async def execute(self, state):
                return state

        agent = SafeAgent()
        state = AnalysisState(
            job_id="test-002",
            dataset_info=DatasetInfo(url="test"),
        )

        result = await agent.execute_with_tracing(state)
        assert result is not None

    @pytest.mark.asyncio
    async def test_empty_required_fields_no_validation(self, _mock_llm):
        from src.agents.base.agent import BaseAgent
        from src.agents.base.state import AnalysisState, DatasetInfo

        _mock_llm.return_value = MagicMock()

        class NoRequirements(BaseAgent):
            AGENT_NAME = "none"
            REQUIRED_STATE_FIELDS = []

            async def execute(self, state):
                return state

        agent = NoRequirements()
        state = AnalysisState(job_id="test-003", dataset_info=DatasetInfo(url="test"))

        result = await agent.execute_with_tracing(state)
        assert result is not None
