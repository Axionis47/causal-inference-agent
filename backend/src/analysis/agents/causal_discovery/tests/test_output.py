"""Validation tests for CausalDAG, CausalEdge, and CausalPair typed models."""

import pytest
from pydantic import ValidationError

from src.analysis.agents.causal_discovery import CausalDAG, CausalEdge, CausalPair


class TestCausalEdge:
    def test_defaults(self):
        e = CausalEdge(source="a", target="b")
        assert e.edge_type == "directed"
        assert e.confidence == 1.0

    def test_full_construction(self):
        e = CausalEdge(source="a", target="b", edge_type="undirected", confidence=0.7)
        assert e.edge_type == "undirected"
        assert e.confidence == 0.7

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            CausalEdge(target="b")


class TestCausalPair:
    def test_defaults(self):
        p = CausalPair(treatment="t", outcome="y")
        assert p.priority == 1
        assert p.rationale == ""

    def test_full_construction(self):
        p = CausalPair(treatment="t", outcome="y", rationale="primary hypothesis", priority=2)
        assert p.rationale == "primary hypothesis"
        assert p.priority == 2


class TestCausalDAG:
    def test_minimum_required_fields(self):
        d = CausalDAG(nodes=["a", "b"], edges=[], discovery_method="test")
        assert d.treatment_variable is None
        assert d.outcome_variable is None
        assert d.interpretation == ""
        assert d.forbidden_edges is None
        assert d.variable_roles is None
        assert d.adjustment_set is None

    def test_full_construction(self):
        d = CausalDAG(
            nodes=["t", "y", "x"],
            edges=[CausalEdge(source="t", target="y"), CausalEdge(source="x", target="y")],
            discovery_method="PC",
            treatment_variable="t",
            outcome_variable="y",
            interpretation="t -> y, x confounds y",
            adjustment_set=["x"],
        )
        assert d.adjustment_set == ["x"]
        assert d.treatment_variable == "t"

    def test_round_trip(self):
        """model_dump / model_validate must round-trip cleanly for persistence."""
        d1 = CausalDAG(
            nodes=["t", "y"],
            edges=[CausalEdge(source="t", target="y", confidence=0.9)],
            discovery_method="PC",
            treatment_variable="t",
            outcome_variable="y",
        )
        d2 = CausalDAG.model_validate(d1.model_dump())
        assert d2 == d1
