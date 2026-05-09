"""Tests for the discovered_dag / refined_dag split.

Before this change, both causal_discovery and dag_expert wrote to a
single proposed_dag field, so the second silently overwrote the first.
The fields are now separate; proposed_dag is a read-through computed
property that prefers refined_dag when available.
"""

from __future__ import annotations

import pytest

from src.agents import AnalysisState, DatasetInfo
from src.agents.base.state import CausalDAG, CausalEdge


def _state() -> AnalysisState:
    return AnalysisState(
        job_id="t",
        dataset_info=DatasetInfo(url="u", name="n"),
        treatment_variable="treat",
        outcome_variable="outcome",
    )


def _dag(method: str, edges: list[tuple[str, str]] | None = None) -> CausalDAG:
    return CausalDAG(
        nodes=["a", "b", "c"],
        edges=[CausalEdge(source=s, target=t, edge_type="directed")
               for s, t in (edges or [("a", "b")])],
        discovery_method=method,
    )


class TestDagFieldsIndependent:
    """discovered_dag and refined_dag are separate slots."""

    def test_neither_set(self):
        s = _state()
        assert s.discovered_dag is None
        assert s.refined_dag is None
        assert s.proposed_dag is None

    def test_only_discovered_set(self):
        s = _state()
        s.discovered_dag = _dag("PC")
        assert s.discovered_dag is not None
        assert s.refined_dag is None
        # proposed_dag falls back to discovered_dag
        assert s.proposed_dag is s.discovered_dag

    def test_only_refined_set(self):
        s = _state()
        s.refined_dag = _dag("dag_expert")
        assert s.discovered_dag is None
        assert s.refined_dag is not None
        assert s.proposed_dag is s.refined_dag

    def test_both_set_refined_wins(self):
        s = _state()
        discovered = _dag("PC", [("a", "b")])
        refined = _dag("domain_expert_fusion", [("a", "b"), ("b", "c")])
        s.discovered_dag = discovered
        s.refined_dag = refined
        # Both preserved, neither overwrites the other
        assert s.discovered_dag is discovered
        assert s.refined_dag is refined
        # proposed_dag prefers refined
        assert s.proposed_dag is refined


class TestProposedDagIsReadOnly:
    """proposed_dag is a computed property; direct writes raise."""

    def test_setting_proposed_dag_raises(self):
        s = _state()
        with pytest.raises((AttributeError, ValueError)):
            s.proposed_dag = _dag("any")  # type: ignore[misc]


class TestLegacyMigration:
    """A legacy state doc with proposed_dag is loaded into refined_dag."""

    def test_legacy_proposed_dag_becomes_refined(self):
        loaded = AnalysisState.model_validate({
            "job_id": "legacy",
            "dataset_info": {"url": "u", "name": "n"},
            "proposed_dag": {
                "nodes": ["a", "b"],
                "edges": [{"source": "a", "target": "b", "edge_type": "directed"}],
                "discovery_method": "legacy",
            },
        })
        assert loaded.refined_dag is not None
        assert loaded.refined_dag.discovery_method == "legacy"
        # Discovery slot stays empty since we don't know which stage produced it
        assert loaded.discovered_dag is None
        # proposed_dag computed property still resolves
        assert loaded.proposed_dag is loaded.refined_dag

    def test_explicit_new_fields_take_precedence(self):
        # If both old and new keys are present, the new schema wins
        loaded = AnalysisState.model_validate({
            "job_id": "migrating",
            "dataset_info": {"url": "u", "name": "n"},
            "discovered_dag": {
                "nodes": ["a"],
                "edges": [],
                "discovery_method": "PC",
            },
            "refined_dag": {
                "nodes": ["a"],
                "edges": [],
                "discovery_method": "domain_expert_fusion",
            },
            "proposed_dag": {
                "nodes": ["x"],
                "edges": [],
                "discovery_method": "should_be_dropped",
            },
        })
        assert loaded.discovered_dag.discovery_method == "PC"
        assert loaded.refined_dag.discovery_method == "domain_expert_fusion"


class TestSerialization:
    """Both new fields and the computed property survive a round-trip."""

    def test_round_trip_preserves_both(self):
        s = _state()
        s.discovered_dag = _dag("PC")
        s.refined_dag = _dag("domain_expert_fusion")
        dumped = s.model_dump()
        assert dumped["discovered_dag"]["discovery_method"] == "PC"
        assert dumped["refined_dag"]["discovery_method"] == "domain_expert_fusion"
        # proposed_dag is included as a computed field
        assert "proposed_dag" in dumped
        assert dumped["proposed_dag"]["discovery_method"] == "domain_expert_fusion"
