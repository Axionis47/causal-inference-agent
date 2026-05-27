"""Tool-handler tests covering the propose / forbid / fuse / adjustment paths."""

import pytest

from src.analysis.agents.base import ToolResultStatus


class TestToolAnalyzeDomain:
    @pytest.mark.asyncio
    async def test_returns_healthcare_patterns(self, agent, state):
        result = await agent._tool_analyze_domain(state)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["domain"] == "healthcare"
        assert any("Risk factors" in p for p in result.output["typical_patterns"])

    @pytest.mark.asyncio
    async def test_unknown_domain_gets_default_patterns(self, agent, state):
        state.dataset_info.kaggle_domain = None
        result = await agent._tool_analyze_domain(state)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["domain"] == "unknown"
        assert any("Pre-treatment" in p for p in result.output["typical_patterns"])


class TestToolProposeEdge:
    @pytest.mark.asyncio
    async def test_appends_domain_edge(self, agent, state):
        result = await agent._tool_propose_edge(
            state, source="age", target="treat", reasoning="demographics drive selection", confidence="high"
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["total_domain_edges"] == 1
        assert agent._domain_edges[0]["source"] == "age"

    @pytest.mark.asyncio
    async def test_rejects_forbidden_edge(self, agent, state):
        """Edges marked forbidden cannot be proposed afterwards."""
        await agent._tool_mark_forbidden_edge(
            state, source="outcome", target="treat", reason="reverse causation"
        )
        result = await agent._tool_propose_edge(
            state, source="outcome", target="treat", reasoning="x", confidence="low"
        )
        assert result.status == ToolResultStatus.ERROR


class TestToolMarkForbiddenEdge:
    @pytest.mark.asyncio
    async def test_records_forbidden_pair_with_reason(self, agent, state):
        result = await agent._tool_mark_forbidden_edge(
            state, source="outcome", target="treat", reason="reverse causation impossible"
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert agent._forbidden_edges == [("outcome", "treat", "reverse causation impossible")]


class TestToolGetDiscoveryEdges:
    @pytest.mark.asyncio
    async def test_returns_unavailable_when_no_discovery(self, agent, state):
        result = await agent._tool_get_discovery_edges(state)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["available"] is False

    @pytest.mark.asyncio
    async def test_reads_discovered_dag(self, agent, state_with_discovered_dag):
        result = await agent._tool_get_discovery_edges(state_with_discovered_dag)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_edges"] == 3
        assert len(agent._data_edges) == 3


class TestToolFuseAndValidate:
    @pytest.mark.asyncio
    async def test_writes_refined_dag(self, agent, state):
        agent._domain_edges = [
            {"source": "age", "target": "treat", "confidence": "high"},
            {"source": "treat", "target": "outcome", "confidence": "high"},
        ]
        result = await agent._tool_fuse_and_validate(state, conflict_resolution="domain_priority")
        assert result.status == ToolResultStatus.SUCCESS
        assert state.refined_dag is not None
        assert state.refined_dag.discovery_method == "domain_expert_fusion (domain_priority)"
        assert len(state.refined_dag.edges) == 2

    @pytest.mark.asyncio
    async def test_drops_forbidden_edges(self, agent, state):
        agent._domain_edges = [{"source": "outcome", "target": "treat", "confidence": "low"}]
        agent._forbidden_edges = [("outcome", "treat", "reverse causation")]
        await agent._tool_fuse_and_validate(state, conflict_resolution="domain_priority")
        assert state.refined_dag is not None
        assert len(state.refined_dag.edges) == 0

    @pytest.mark.asyncio
    async def test_persists_variable_roles_and_forbidden(self, agent, state):
        agent._variable_roles = {"age": "confounder", "treat": "treatment"}
        agent._forbidden_edges = [("outcome", "treat", "reverse")]
        agent._domain_edges = [{"source": "treat", "target": "outcome", "confidence": "high"}]
        await agent._tool_fuse_and_validate(state, conflict_resolution="domain_priority")
        assert state.refined_dag.variable_roles == {"age": "confounder", "treat": "treatment"}
        assert state.refined_dag.forbidden_edges == [
            {"source": "outcome", "target": "treat", "reason": "reverse"}
        ]


class TestToolGetAdjustmentSet:
    @pytest.mark.asyncio
    async def test_errors_without_refined_dag(self, agent, state):
        result = await agent._tool_get_adjustment_set(state, treatment="treat", outcome="outcome")
        assert result.status == ToolResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_identifies_confounder_and_writes_adjustment_set(self, agent, state):
        """age -> treat, age -> outcome, treat -> outcome: adjustment set is [age]."""
        agent._domain_edges = [
            {"source": "age", "target": "treat", "confidence": "high"},
            {"source": "age", "target": "outcome", "confidence": "high"},
            {"source": "treat", "target": "outcome", "confidence": "high"},
        ]
        await agent._tool_fuse_and_validate(state, conflict_resolution="domain_priority")
        result = await agent._tool_get_adjustment_set(state, treatment="treat", outcome="outcome")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["adjustment_set"] == ["age"]
        assert state.refined_dag.adjustment_set == ["age"]
