"""Shared fixtures for dag_expert tests."""

import pytest

from src.analysis.agents.base import AnalysisState, CausalDAG, CausalEdge, DatasetInfo
from src.analysis.agents.dag_expert.agent import DAGExpertAgent


@pytest.fixture
def agent():
    return DAGExpertAgent()


@pytest.fixture
def state():
    """State with a treatment/outcome pair plus a kaggle_domain hint."""
    return AnalysisState(
        job_id="dag-job",
        dataset_info=DatasetInfo(
            url="https://kaggle.com/x/y",
            name="d",
            kaggle_domain="healthcare",
            kaggle_tags=["medicine", "outcomes"],
        ),
        treatment_variable="treat",
        outcome_variable="outcome",
    )


@pytest.fixture
def state_with_discovered_dag(state):
    """State that has a discovered_dag set, so get_discovery_edges has something to read."""
    state.discovered_dag = CausalDAG(
        nodes=["treat", "age", "outcome"],
        edges=[
            CausalEdge(source="age", target="treat", edge_type="directed", confidence=0.8),
            CausalEdge(source="age", target="outcome", edge_type="directed", confidence=0.8),
            CausalEdge(source="treat", target="outcome", edge_type="directed", confidence=0.9),
        ],
        discovery_method="PC (test)",
        treatment_variable="treat",
        outcome_variable="outcome",
    )
    return state
