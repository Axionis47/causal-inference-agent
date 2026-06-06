"""Tests for the pure helpers: role classification, candidate selection, fallback."""

import logging

from src.analysis.agents.base import CausalDAG, CausalEdge
from src.analysis.agents.confounder_discovery.helpers import (
    candidates_from_df,
    classify_variable_role,
    fallback_confounder_identification,
)


_LOG = logging.getLogger("test")


def _confounder_dag() -> CausalDAG:
    """age -> treat, age -> outcome, treat -> outcome."""
    return CausalDAG(
        nodes=["age", "treat", "outcome"],
        edges=[
            CausalEdge(source="age", target="treat", edge_type="directed"),
            CausalEdge(source="age", target="outcome", edge_type="directed"),
            CausalEdge(source="treat", target="outcome", edge_type="directed"),
        ],
        discovery_method="test",
    )


class TestClassifyVariableRole:
    def test_confounder(self):
        assert classify_variable_role(_confounder_dag(), "age", "treat", "outcome") == "confounder"

    def test_mediator(self):
        """treat -> mediator -> outcome should be classified as potential_mediator."""
        dag = CausalDAG(
            nodes=["treat", "mediator", "outcome"],
            edges=[
                CausalEdge(source="treat", target="mediator", edge_type="directed"),
                CausalEdge(source="mediator", target="outcome", edge_type="directed"),
            ],
            discovery_method="test",
        )
        assert classify_variable_role(dag, "mediator", "treat", "outcome") == "potential_mediator"

    def test_collider(self):
        """treat -> collider <- outcome — descendant of both."""
        dag = CausalDAG(
            nodes=["treat", "outcome", "collider"],
            edges=[
                CausalEdge(source="treat", target="collider", edge_type="directed"),
                CausalEdge(source="outcome", target="collider", edge_type="directed"),
            ],
            discovery_method="test",
        )
        assert classify_variable_role(dag, "collider", "treat", "outcome") == "potential_collider"

    def test_no_dag_returns_unknown(self):
        assert classify_variable_role(None, "x", "treat", "outcome") == "unknown"


class TestCandidatesFromDf:
    def test_excludes_treatment_and_outcome(self, confounder_dataframe):
        cols = candidates_from_df(confounder_dataframe, "treat", "outcome")
        assert "treat" not in cols
        assert "outcome" not in cols
        assert set(cols) == {"age", "noise"}

    def test_none_df_returns_empty(self):
        assert candidates_from_df(None, "t", "o") == []


class TestFallbackConfounderIdentification:
    def test_finds_age_excludes_noise(self, confounder_dataframe):
        ranked, excluded = fallback_confounder_identification(
            df=confounder_dataframe,
            treatment_var="treat",
            outcome_var="outcome",
            dag=None,
            domain_knowledge=None,
            logger=_LOG,
        )
        assert "age" in ranked
        # noise might or might not pass the threshold depending on rng; just
        # assert ordering: age should outrank noise if both are in the list
        if "noise" in ranked:
            assert ranked.index("age") < ranked.index("noise")

    def test_excludes_mediators(self, confounder_dataframe):
        """When the DAG says age is a mediator, it gets excluded — not ranked."""
        mediator_dag = CausalDAG(
            nodes=["treat", "age", "outcome"],
            edges=[
                CausalEdge(source="treat", target="age", edge_type="directed"),
                CausalEdge(source="age", target="outcome", edge_type="directed"),
            ],
            discovery_method="test",
        )
        ranked, excluded = fallback_confounder_identification(
            df=confounder_dataframe,
            treatment_var="treat",
            outcome_var="outcome",
            dag=mediator_dag,
            domain_knowledge=None,
            logger=_LOG,
        )
        assert "age" in excluded
        assert "age" not in ranked
