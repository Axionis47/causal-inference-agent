"""Unit tests for CausalDiscoveryAgent (ReAct-based)."""

import pytest
import pandas as pd
import numpy as np

from src.agents.base import (
    AnalysisState,
    DatasetInfo,
    DataProfile,
    CausalDAG,
    CausalEdge,
    ToolResultStatus,
)
from src.agents.specialists.causal_discovery import CausalDiscoveryAgent


@pytest.fixture
def agent():
    """Create agent instance."""
    return CausalDiscoveryAgent()


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame({
        "treat": np.random.binomial(1, 0.4, n),
        "age": np.random.normal(40, 10, n),
        "income": np.random.normal(50000, 15000, n),
        "education": np.random.randint(10, 20, n),
        "outcome": np.random.normal(100, 20, n),
    })


@pytest.fixture
def sample_profile(sample_dataframe):
    """Create a sample data profile."""
    profile = DataProfile(
        n_samples=100,
        n_features=5,
        feature_names=list(sample_dataframe.columns),
        feature_types={
            "treat": "binary",
            "age": "numeric",
            "income": "numeric",
            "education": "numeric",
            "outcome": "numeric",
        },
        missing_values={col: 0 for col in sample_dataframe.columns},
        numeric_stats={
            "treat": {"mean": 0.4, "std": 0.49, "min": 0.0, "max": 1.0},
            "age": {"mean": 40.0, "std": 10.0, "min": 20.0, "max": 60.0},
            "income": {"mean": 50000.0, "std": 15000.0, "min": 20000.0, "max": 80000.0},
            "education": {"mean": 15.0, "std": 3.0, "min": 10.0, "max": 20.0},
            "outcome": {"mean": 100.0, "std": 20.0, "min": 60.0, "max": 140.0},
        },
        categorical_stats={},
        treatment_candidates=["treat"],
        outcome_candidates=["outcome", "income"],
        potential_confounders=["age", "education"],
    )
    return profile


@pytest.fixture
def state_with_dataframe(sample_dataframe, sample_profile, tmp_path):
    """Create state with a loaded dataframe."""
    import pickle

    # Save dataframe to temp path
    pkl_path = tmp_path / "test_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(sample_dataframe, f)

    return AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(
            url="https://kaggle.com/test/dataset",
            name="test_dataset",
            local_path=str(pkl_path),
            n_samples=100,
            n_features=5,
        ),
        dataframe_path=str(pkl_path),
        data_profile=sample_profile,
    )


class TestInitialization:
    """Test agent initialization."""

    def test_agent_name(self, agent):
        """Test agent has correct name."""
        assert agent.AGENT_NAME == "causal_discovery"

    def test_max_steps(self, agent):
        """Test agent has reasonable max steps."""
        assert agent.MAX_STEPS == 15

    def test_tools_registered(self, agent):
        """Test that discovery tools are registered."""
        tool_names = list(agent._tools.keys())

        expected_tools = [
            # Discovery tools
            "get_data_characteristics",
            "run_discovery_algorithm",
            "inspect_graph",
            "validate_graph",
            "compare_algorithms",
            "finalize_discovery",
            # Context tools from mixin
            "ask_domain_knowledge",
            "get_column_info",
            "get_dataset_summary",
            "list_columns",
            # Built-in ReAct tools
            "finish",
            "reflect",
        ]

        for tool in expected_tools:
            assert tool in tool_names, f"Tool '{tool}' not registered"

    def test_has_context_tools(self, agent):
        """Test that context tools mixin is properly integrated."""
        assert "ask_domain_knowledge" in agent._tools
        assert "get_column_info" in agent._tools


class TestToolGetDataCharacteristics:
    """Test get_data_characteristics tool."""

    @pytest.mark.asyncio
    async def test_returns_characteristics(self, agent, sample_dataframe, state_with_dataframe):
        """Test getting data characteristics."""
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
        """Test that treatment/outcome are included."""
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"

        result = await agent._tool_get_data_characteristics(state_with_dataframe)

        assert result.output["treatment"] == "treat"
        assert result.output["outcome"] == "outcome"

    @pytest.mark.asyncio
    async def test_error_when_no_dataframe(self, agent, state_with_dataframe):
        """Test error when dataframe not loaded."""
        agent._df = None

        result = await agent._tool_get_data_characteristics(state_with_dataframe)

        assert result.status == ToolResultStatus.ERROR
        assert "not loaded" in result.error


class TestToolRunDiscoveryAlgorithm:
    """Test run_discovery_algorithm tool."""

    @pytest.mark.asyncio
    async def test_runs_algorithm(self, agent, sample_dataframe, state_with_dataframe):
        """Test running a discovery algorithm."""
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        agent._current_state = state_with_dataframe

        result = await agent._tool_run_discovery_algorithm(
            state_with_dataframe, algorithm="pc"
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert "algorithm" in result.output
        assert "n_nodes" in result.output
        assert "n_edges" in result.output

    @pytest.mark.asyncio
    async def test_stores_graph(self, agent, sample_dataframe, state_with_dataframe):
        """Test that discovered graph is stored."""
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        agent._current_state = state_with_dataframe

        await agent._tool_run_discovery_algorithm(
            state_with_dataframe, algorithm="pc"
        )

        assert "pc" in agent._discovered_graphs or agent._current_graph is not None

    @pytest.mark.asyncio
    async def test_error_when_no_dataframe(self, agent, state_with_dataframe):
        """Test error when dataframe not loaded."""
        agent._df = None

        result = await agent._tool_run_discovery_algorithm(
            state_with_dataframe, algorithm="pc"
        )

        assert result.status == ToolResultStatus.ERROR


class TestToolInspectGraph:
    """Test inspect_graph tool."""

    @pytest.mark.asyncio
    async def test_inspects_graph(self, agent, state_with_dataframe):
        """Test inspecting a graph."""
        # Create a test graph
        agent._current_graph = CausalDAG(
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
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"

        result = await agent._tool_inspect_graph(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_nodes"] == 3
        assert result.output["n_directed"] == 3

    @pytest.mark.asyncio
    async def test_identifies_confounders(self, agent, state_with_dataframe):
        """Test that confounders are identified."""
        # Create a graph with a confounder
        agent._current_graph = CausalDAG(
            nodes=["treat", "age", "outcome"],
            edges=[
                CausalEdge(source="age", target="treat", edge_type="directed"),
                CausalEdge(source="age", target="outcome", edge_type="directed"),
                CausalEdge(source="treat", target="outcome", edge_type="directed"),
            ],
            discovery_method="test",
        )
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"

        result = await agent._tool_inspect_graph(state_with_dataframe)

        assert "age" in result.output["potential_confounders"]

    @pytest.mark.asyncio
    async def test_error_when_no_graph(self, agent, state_with_dataframe):
        """Test error when no graph exists."""
        agent._current_graph = None

        result = await agent._tool_inspect_graph(state_with_dataframe)

        assert result.status == ToolResultStatus.ERROR


class TestToolValidateGraph:
    """Test validate_graph tool."""

    @pytest.mark.asyncio
    async def test_validates_good_graph(self, agent, state_with_dataframe):
        """Test validating a reasonable graph."""
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
        assert result.output["has_treatment_outcome_edge"] == True

    @pytest.mark.asyncio
    async def test_detects_reverse_causation(self, agent, state_with_dataframe):
        """Test that reverse causation is detected."""
        agent._current_graph = CausalDAG(
            nodes=["treat", "outcome"],
            edges=[
                CausalEdge(source="outcome", target="treat", edge_type="directed"),
            ],
            discovery_method="test",
        )
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"

        result = await agent._tool_validate_graph(state_with_dataframe)

        assert result.output["reverse_causation"] == True
        assert len(result.output["issues"]) > 0

    @pytest.mark.asyncio
    async def test_error_when_no_graph(self, agent, state_with_dataframe):
        """Test error when no graph exists."""
        agent._current_graph = None

        result = await agent._tool_validate_graph(state_with_dataframe)

        assert result.status == ToolResultStatus.ERROR


class TestToolCompareAlgorithms:
    """Test compare_algorithms tool."""

    @pytest.mark.asyncio
    async def test_compares_algorithms(self, agent, state_with_dataframe):
        """Test comparing multiple algorithms."""
        # Create two graphs
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
        """Test message when only one algorithm run."""
        agent._discovered_graphs = {
            "pc": CausalDAG(
                nodes=["treat", "outcome"],
                edges=[],
                discovery_method="PC",
            ),
        }

        result = await agent._tool_compare_algorithms(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_algorithms"] == 1
        assert "Run more" in result.output["message"]


class TestToolFinalizeDiscovery:
    """Test finalize_discovery tool."""

    @pytest.mark.asyncio
    async def test_finalizes_discovery(self, agent, state_with_dataframe):
        """Test finalizing the discovery."""
        result = await agent._tool_finalize_discovery(
            state_with_dataframe,
            chosen_algorithm="pc",
            interpretation="Treatment causes outcome with age as confounder",
            confidence="high",
            confounders=["age"],
            mediators=[],
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["discovery_finalized"] == True
        assert agent._finalized == True
        assert agent._final_result["chosen_algorithm"] == "pc"
        assert agent._final_result["confidence"] == "high"


class TestCreateSimpleDag:
    """Test simple DAG creation."""

    def test_creates_simple_dag(self, agent, state_with_dataframe):
        """Test creating a simple DAG."""
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        agent._current_state = state_with_dataframe

        dag = agent._create_simple_dag()

        assert "treat" in dag.nodes
        assert "outcome" in dag.nodes
        assert any(
            e.source == "treat" and e.target == "outcome"
            for e in dag.edges
        )

    def test_includes_confounders(self, agent, state_with_dataframe):
        """Test that confounders are added."""
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        agent._current_state = state_with_dataframe

        dag = agent._create_simple_dag()

        # Should have confounders from profile
        assert len(dag.nodes) > 2


class TestAutoFinalize:
    """Test auto-finalization when agent doesn't call finalize."""

    def test_auto_finalize_with_graph(self, agent):
        """Test auto-finalize when a graph was discovered."""
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
        """Test auto-finalize when no graph was discovered."""
        agent._discovered_graphs = {}

        result = agent._auto_finalize()

        assert result["chosen_algorithm"] == "simple"
        assert result["confidence"] == "low"


class TestCheckPath:
    """Test path checking utility."""

    def test_finds_direct_path(self, agent):
        """Test finding a direct path."""
        dag = CausalDAG(
            nodes=["a", "b"],
            edges=[CausalEdge(source="a", target="b", edge_type="directed")],
            discovery_method="test",
        )

        assert agent._check_path(dag, "a", "b") == True

    def test_finds_indirect_path(self, agent):
        """Test finding an indirect path."""
        dag = CausalDAG(
            nodes=["a", "b", "c"],
            edges=[
                CausalEdge(source="a", target="b", edge_type="directed"),
                CausalEdge(source="b", target="c", edge_type="directed"),
            ],
            discovery_method="test",
        )

        assert agent._check_path(dag, "a", "c") == True

    def test_no_path(self, agent):
        """Test when no path exists."""
        dag = CausalDAG(
            nodes=["a", "b"],
            edges=[CausalEdge(source="b", target="a", edge_type="directed")],
            discovery_method="test",
        )

        assert agent._check_path(dag, "a", "b") == False


class TestInitialObservation:
    """Test initial observation generation."""

    def test_generates_lean_observation(self, agent, state_with_dataframe):
        """Test that initial observation is lean."""
        obs = agent._get_initial_observation(state_with_dataframe)

        # Should be short
        assert len(obs) < 600

        # Should mention key elements
        assert "Treatment" in obs or "treatment" in obs
        assert "Outcome" in obs or "outcome" in obs


class TestTaskCompletion:
    """Test task completion logic."""

    @pytest.mark.asyncio
    async def test_not_complete_initially(self, agent, state_with_dataframe):
        """Test that task is not complete initially."""
        result = await agent.is_task_complete(state_with_dataframe)
        assert result == False

    @pytest.mark.asyncio
    async def test_complete_after_finalize(self, agent, state_with_dataframe):
        """Test task is complete after finalize."""
        # Finalize
        await agent._tool_finalize_discovery(
            state_with_dataframe,
            chosen_algorithm="pc",
            interpretation="Test",
            confidence="high",
        )

        # Set DAG on state
        state_with_dataframe.proposed_dag = CausalDAG(
            nodes=["a", "b"],
            edges=[],
            discovery_method="test",
        )

        result = await agent.is_task_complete(state_with_dataframe)
        assert result == True


class TestContextToolsIntegration:
    """Test that context tools work with the discovery agent."""

    @pytest.mark.asyncio
    async def test_ask_domain_knowledge_available(self, agent, state_with_dataframe):
        """Test that ask_domain_knowledge tool is callable."""
        result = await agent._ask_domain_knowledge(
            state_with_dataframe,
            question="What is the treatment variable?"
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["found"] == False  # No domain knowledge set

    @pytest.mark.asyncio
    async def test_list_columns_with_profile(self, agent, state_with_dataframe):
        """Test list_columns with data profile."""
        result = await agent._list_columns(state_with_dataframe)

        assert result.status == ToolResultStatus.SUCCESS
        assert "treat" in result.output["columns"]
