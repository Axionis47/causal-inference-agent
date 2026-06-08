"""Tests for the pure helpers: patterns_for_domain, fuse_edges, adjustment_set_from_dag."""

import types

from src.analysis.agents.base import CausalDAG, CausalEdge
from src.analysis.agents.dag_expert.helpers import (
    adjustment_set_from_dag,
    build_fallback_dag,
    fuse_edges,
    patterns_for_domain,
)


def _state(treatment, outcome, roles=None, confounders=None):
    profile = types.SimpleNamespace(potential_confounders=confounders or [])
    return types.SimpleNamespace(
        job_id="t",
        treatment_variable=treatment,
        outcome_variable=outcome,
        data_profile=profile,
    )


class TestBuildFallbackDag:
    def test_uses_classified_confounder_roles(self):
        roles = {
            "treat": "treatment",
            "re78": "outcome",
            "age": "confounder",
            "educ": "confounder",
        }
        dag = build_fallback_dag(_state("treat", "re78", roles), roles)
        assert set(dag.adjustment_set) == {"age", "educ"}
        # canonical edges: each confounder -> T and -> Y, plus T -> Y
        pairs = {(e.source, e.target) for e in dag.edges}
        assert ("age", "treat") in pairs and ("age", "re78") in pairs
        assert ("treat", "re78") in pairs

    def test_falls_back_to_profiler_confounders_when_no_roles(self):
        dag = build_fallback_dag(
            _state("treat", "re78", roles={}, confounders=["age", "re74"]),
            {},
        )
        assert set(dag.adjustment_set) == {"age", "re74"}

    def test_never_includes_treatment_or_outcome_in_adjustment_set(self):
        roles = {"treat": "treatment", "re78": "outcome", "age": "confounder"}
        dag = build_fallback_dag(_state("treat", "re78", roles), roles)
        assert "treat" not in dag.adjustment_set
        assert "re78" not in dag.adjustment_set


class TestPatternsForDomain:
    def test_healthcare(self):
        assert any("Risk factors" in p for p in patterns_for_domain("healthcare"))

    def test_economics(self):
        assert any("Earnings" in p for p in patterns_for_domain("economics"))

    def test_unknown_falls_back_to_default(self):
        out = patterns_for_domain(None)
        assert any("Pre-treatment" in p for p in out)


class TestFuseEdges:
    def test_domain_priority_keeps_domain_direction(self):
        """Domain says a->b, data says b->a, with domain_priority we keep a->b."""
        dag, conflicts, sources = fuse_edges(
            domain_edges=[{"source": "a", "target": "b", "confidence": "high"}],
            data_edges=[{"source": "b", "target": "a", "confidence": 0.8, "edge_type": "directed"}],
            forbidden_edges=[],
            conflict_resolution="domain_priority",
            treatment_var=None,
            outcome_var=None,
            variable_roles={},
        )
        assert len(conflicts) == 1
        assert any(e.source == "a" and e.target == "b" for e in dag.edges)
        assert not any(e.source == "b" and e.target == "a" for e in dag.edges)

    def test_consensus_only_skips_domain_keeps_data(self):
        """consensus_only drops the domain copy of a conflict but still adds the data edge.

        Documented quirk of the existing monolith: the domain loop continues when
        reverse_key is in data_edge_set, but the data loop unconditionally adds
        the data edge since the reverse pair was never processed. Net effect is
        that data wins, not that the conflict is dropped.
        """
        dag, conflicts, _ = fuse_edges(
            domain_edges=[{"source": "a", "target": "b", "confidence": "high"}],
            data_edges=[{"source": "b", "target": "a", "confidence": 0.8, "edge_type": "directed"}],
            forbidden_edges=[],
            conflict_resolution="consensus_only",
            treatment_var=None,
            outcome_var=None,
            variable_roles={},
        )
        assert len(conflicts) == 1
        assert any(e.source == "b" and e.target == "a" for e in dag.edges)
        assert not any(e.source == "a" and e.target == "b" for e in dag.edges)

    def test_forbidden_filters_both_domain_and_data(self):
        dag, _, _ = fuse_edges(
            domain_edges=[{"source": "outcome", "target": "treat", "confidence": "low"}],
            data_edges=[{"source": "outcome", "target": "treat", "confidence": 0.6, "edge_type": "directed"}],
            forbidden_edges=[("outcome", "treat", "reverse causation")],
            conflict_resolution="domain_priority",
            treatment_var="treat",
            outcome_var="outcome",
            variable_roles={},
        )
        assert dag.edges == []


class TestAdjustmentSetFromDag:
    def test_simple_backdoor(self):
        """Classic confounder pattern: age -> {treat, outcome}, treat -> outcome."""
        dag = CausalDAG(
            nodes=["age", "treat", "outcome"],
            edges=[
                CausalEdge(source="age", target="treat", edge_type="directed"),
                CausalEdge(source="age", target="outcome", edge_type="directed"),
                CausalEdge(source="treat", target="outcome", edge_type="directed"),
            ],
            discovery_method="test",
        )
        result = adjustment_set_from_dag(dag, "treat", "outcome")
        assert result["adjustment_set"] == ["age"]
        assert result["confounders"] == ["age"]
        assert result["mediators"] == []

    def test_mediator_excluded_from_adjustment(self):
        """treat -> mediator -> outcome: never adjust for the mediator."""
        dag = CausalDAG(
            nodes=["treat", "mediator", "outcome"],
            edges=[
                CausalEdge(source="treat", target="mediator", edge_type="directed"),
                CausalEdge(source="mediator", target="outcome", edge_type="directed"),
            ],
            discovery_method="test",
        )
        result = adjustment_set_from_dag(dag, "treat", "outcome")
        assert result["adjustment_set"] == []
        assert "mediator" in result["mediators"]
