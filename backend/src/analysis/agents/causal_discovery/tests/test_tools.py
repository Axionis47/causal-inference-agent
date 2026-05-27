"""Tool-handler tests: one class per registered discovery tool."""

import pytest

from src.analysis.agents.base import CausalDAG, CausalEdge, ToolResultStatus


def _three_node_confounded_dag() -> CausalDAG:
    """age -> treat, age -> outcome, treat -> outcome — used in inspect/validate tests."""
    return CausalDAG(
        nodes=["treat", "age", "outcome"],
        edges=[
            CausalEdge(source="age", target="treat", edge_type="directed"),
            CausalEdge(source="age", target="outcome", edge_type="directed"),
            CausalEdge(source="treat", target="outcome", edge_type="directed"),
        ],
        discovery_method="test",
        treatment_variable="treat",
        outcome_variable="outcome",
    )


class TestToolGetDataCharacteristics:
    @pytest.mark.asyncio
    async def test_returns_characteristics(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        result = await agent._tool_get_data_characteristics(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert "n_samples" in result.output
        assert "n_variables" in result.output
        assert "distributions" in result.output
        assert "recommendations" in result.output

    @pytest.mark.asyncio
    async def test_includes_treatment_outcome(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        result = await agent._tool_get_data_characteristics(state_with_dataframe)
        assert result.output["treatment"] == "treat"
        assert result.output["outcome"] == "outcome"

    @pytest.mark.asyncio
    async def test_error_when_no_dataframe(self, agent, state_with_dataframe):
        agent._df = None
        result = await agent._tool_get_data_characteristics(state_with_dataframe)
        assert result.status == ToolResultStatus.ERROR
        assert "not loaded" in result.error


class TestToolRunDiscoveryAlgorithm:
    @pytest.mark.asyncio
    async def test_runs_algorithm(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        agent._current_state = state_with_dataframe
        result = await agent._tool_run_discovery_algorithm(state_with_dataframe, algorithm="pc")
        assert result.status == ToolResultStatus.SUCCESS
        assert "algorithm" in result.output
        assert "n_nodes" in result.output
        assert "n_edges" in result.output

    @pytest.mark.asyncio
    async def test_stores_graph(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        agent._current_state = state_with_dataframe
        await agent._tool_run_discovery_algorithm(state_with_dataframe, algorithm="pc")
        assert "pc" in agent._discovered_graphs or agent._current_graph is not None

    @pytest.mark.asyncio
    async def test_error_when_no_dataframe(self, agent, state_with_dataframe):
        agent._df = None
        result = await agent._tool_run_discovery_algorithm(state_with_dataframe, algorithm="pc")
        assert result.status == ToolResultStatus.ERROR


class TestToolInspectGraph:
    @pytest.mark.asyncio
    async def test_inspects_graph(self, agent, state_with_dataframe):
        agent._current_graph = _three_node_confounded_dag()
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        result = await agent._tool_inspect_graph(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_nodes"] == 3
        assert result.output["n_directed"] == 3

    @pytest.mark.asyncio
    async def test_identifies_confounders(self, agent, state_with_dataframe):
        agent._current_graph = _three_node_confounded_dag()
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        result = await agent._tool_inspect_graph(state_with_dataframe)
        assert "age" in result.output["potential_confounders"]

    @pytest.mark.asyncio
    async def test_error_when_no_graph(self, agent, state_with_dataframe):
        agent._current_graph = None
        result = await agent._tool_inspect_graph(state_with_dataframe)
        assert result.status == ToolResultStatus.ERROR


class TestToolValidateGraph:
    @pytest.mark.asyncio
    async def test_validates_good_graph(self, agent, state_with_dataframe):
        agent._current_graph = CausalDAG(
            nodes=["treat", "age", "outcome"],
            edges=[
                CausalEdge(source="age", target="treat", edge_type="directed"),
                CausalEdge(source="treat", target="outcome", edge_type="directed"),
            ],
            discovery_method="test",
        )
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        result = await agent._tool_validate_graph(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_treatment_outcome_edge"] is True

    @pytest.mark.asyncio
    async def test_detects_reverse_causation(self, agent, state_with_dataframe):
        """outcome -> treat surfaces as both reverse_causation and an issue."""
        agent._current_graph = CausalDAG(
            nodes=["treat", "outcome"],
            edges=[CausalEdge(source="outcome", target="treat", edge_type="directed")],
            discovery_method="test",
        )
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        result = await agent._tool_validate_graph(state_with_dataframe)
        assert result.output["reverse_causation"] is True
        assert len(result.output["issues"]) > 0

    @pytest.mark.asyncio
    async def test_error_when_no_graph(self, agent, state_with_dataframe):
        agent._current_graph = None
        result = await agent._tool_validate_graph(state_with_dataframe)
        assert result.status == ToolResultStatus.ERROR


class TestToolCompareAlgorithms:
    @pytest.mark.asyncio
    async def test_compares_algorithms(self, agent, state_with_dataframe):
        agent._discovered_graphs = {
            "pc": CausalDAG(
                nodes=["treat", "outcome"],
                edges=[CausalEdge(source="treat", target="outcome", edge_type="directed")],
                discovery_method="PC",
            ),
            "ges": CausalDAG(
                nodes=["treat", "outcome"],
                edges=[CausalEdge(source="treat", target="outcome", edge_type="directed")],
                discovery_method="GES",
            ),
        }
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        result = await agent._tool_compare_algorithms(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_algorithms"] == 2
        assert len(result.output["comparison"]) == 2

    @pytest.mark.asyncio
    async def test_message_when_not_enough_algorithms(self, agent, state_with_dataframe):
        agent._discovered_graphs = {
            "pc": CausalDAG(nodes=["treat", "outcome"], edges=[], discovery_method="PC"),
        }
        result = await agent._tool_compare_algorithms(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_algorithms"] == 1
        assert "Run more" in result.output["message"]


class TestToolFinalizeDiscovery:
    @pytest.mark.asyncio
    async def test_finalizes_discovery(self, agent, state_with_dataframe):
        """Finalize records the LLM's choice and flips agent._finalized."""
        result = await agent._tool_finalize_discovery(
            state_with_dataframe,
            chosen_algorithm="pc",
            interpretation="Treatment causes outcome with age as confounder",
            confidence="high",
            confounders=["age"],
            mediators=[],
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["discovery_finalized"] is True
        assert agent._finalized is True
        assert agent._final_result["chosen_algorithm"] == "pc"
        assert agent._final_result["confidence"] == "high"
