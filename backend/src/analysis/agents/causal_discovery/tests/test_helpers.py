"""Tests for the pure helpers: create_simple_dag, auto_finalize, check_path."""

from src.analysis.agents.base import CausalDAG, CausalEdge


class TestCreateSimpleDag:
    def test_creates_simple_dag(self, agent, state_with_dataframe):
        """Fallback DAG always contains treatment, outcome, and a direct edge."""
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        agent._current_state = state_with_dataframe
        dag = agent._create_simple_dag()
        assert "treat" in dag.nodes
        assert "outcome" in dag.nodes
        assert any(e.source == "treat" and e.target == "outcome" for e in dag.edges)

    def test_includes_confounders(self, agent, state_with_dataframe):
        """Profile.potential_confounders gets folded into the fallback DAG nodes."""
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        agent._current_state = state_with_dataframe
        dag = agent._create_simple_dag()
        assert len(dag.nodes) > 2


class TestAutoFinalize:
    def test_auto_finalize_with_graph(self, agent):
        """When a PC graph exists, auto-finalize picks it and identifies confounders."""
        agent._discovered_graphs = {
            "pc": CausalDAG(
                nodes=["treat", "age", "outcome"],
                edges=[
                    CausalEdge(source="age", target="treat", edge_type="directed"),
                    CausalEdge(source="age", target="outcome", edge_type="directed"),
                    CausalEdge(source="treat", target="outcome", edge_type="directed"),
                ],
                discovery_method="PC",
            ),
        }
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        result = agent._auto_finalize()
        assert result["chosen_algorithm"] == "pc"
        assert result["confidence"] == "medium"
        assert "age" in result["confounders"]

    def test_auto_finalize_without_graph(self, agent):
        agent._discovered_graphs = {}
        result = agent._auto_finalize()
        assert result["chosen_algorithm"] == "simple"
        assert result["confidence"] == "low"


class TestCheckPath:
    def test_finds_direct_path(self, agent):
        dag = CausalDAG(
            nodes=["a", "b"],
            edges=[CausalEdge(source="a", target="b", edge_type="directed")],
            discovery_method="test",
        )
        assert agent._check_path(dag, "a", "b") is True

    def test_finds_indirect_path(self, agent):
        dag = CausalDAG(
            nodes=["a", "b", "c"],
            edges=[
                CausalEdge(source="a", target="b", edge_type="directed"),
                CausalEdge(source="b", target="c", edge_type="directed"),
            ],
            discovery_method="test",
        )
        assert agent._check_path(dag, "a", "c") is True

    def test_no_path(self, agent):
        """Reverse edge does not provide a forward path."""
        dag = CausalDAG(
            nodes=["a", "b"],
            edges=[CausalEdge(source="b", target="a", edge_type="directed")],
            discovery_method="test",
        )
        assert agent._check_path(dag, "a", "b") is False
