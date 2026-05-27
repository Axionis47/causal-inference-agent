"""Agent-level tests: initialization, completion, initial observation."""

import pytest

from src.analysis.agents.base import CausalDAG, CausalEdge


class TestInitialization:
    def test_agent_name(self, agent):
        assert agent.AGENT_NAME == "dag_expert"

    def test_max_steps(self, agent):
        assert agent.MAX_STEPS == 12

    def test_tools_registered(self, agent):
        """All 7 dag_expert tools, context-tools mixin, and ReAct built-ins are present."""
        tool_names = list(agent._tools.keys())
        expected = [
            "analyze_domain",
            "classify_variable_role",
            "propose_edge",
            "mark_forbidden_edge",
            "get_discovery_edges",
            "fuse_and_validate",
            "get_adjustment_set",
            "ask_domain_knowledge",
            "get_eda_finding",
            "finish",
            "reflect",
        ]
        for tool in expected:
            assert tool in tool_names, f"Tool '{tool}' not registered"


class TestInitialObservation:
    def test_generates_lean_observation(self, agent, state):
        """Observation names the dataset and a numbered workflow."""
        obs = agent._get_initial_observation(state)
        assert "treat" in obs
        assert "outcome" in obs
        assert "analyze_domain" in obs


class TestTaskCompletion:
    @pytest.mark.asyncio
    async def test_not_complete_initially(self, agent, state):
        """No refined_dag yet means not complete."""
        assert await agent.is_task_complete(state) is False

    @pytest.mark.asyncio
    async def test_complete_requires_roles_and_adjustment(self, agent, state):
        """refined_dag alone isn't enough — variable_roles AND adjustment_set must be set."""
        state.refined_dag = CausalDAG(
            nodes=["treat", "outcome"],
            edges=[CausalEdge(source="treat", target="outcome", edge_type="directed")],
            discovery_method="test",
        )
        assert await agent.is_task_complete(state) is False

        state.refined_dag.variable_roles = {"treat": "treatment", "outcome": "outcome"}
        assert await agent.is_task_complete(state) is False

        state.refined_dag.adjustment_set = []
        assert await agent.is_task_complete(state) is True
