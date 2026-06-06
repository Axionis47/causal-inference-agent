"""Contract tests for the causal_discovery brief.

Pin the capability shape, the preflight refusal cases, and the flag
derivation in build_brief. The ReAct loop and tool behaviour are
exercised by test_agent.py / test_tools.py / test_helpers.py.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.causal_discovery.brief import (
    CAPABILITY,
    build_brief,
    preflight,
)
from src.analysis.agents.causal_discovery.output import CausalDAG, CausalEdge
from src.analysis.agents.data_profiler.output import DataProfile
from src.domain.briefs import Flag


# --- fixtures ---------------------------------------------------------------


def _profile() -> DataProfile:
    return DataProfile(
        n_samples=200,
        n_features=4,
        feature_names=["t", "y", "x1", "x2"],
        feature_types={
            "t": "binary",
            "y": "numeric",
            "x1": "numeric",
            "x2": "numeric",
        },
        missing_values={"t": 0, "y": 0, "x1": 0, "x2": 0},
        numeric_stats={},
        categorical_stats={},
    )


def _make_state(
    *,
    dataframe_path: str | None = "/tmp/df.parquet",
    data_profile: DataProfile | None = None,
    discovered_dag: CausalDAG | None = None,
) -> AnalysisState:
    return AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://example.com/data"),
        dataframe_path=dataframe_path,
        data_profile=data_profile if data_profile is not None else _profile(),
        treatment_variable="t",
        outcome_variable="y",
        discovered_dag=discovered_dag,
    )


def _dag(
    *,
    edges: list[tuple[str, str]] | None = None,
    method: str = "PC (engine)",
    nodes: list[str] | None = None,
) -> CausalDAG:
    edge_list = [
        CausalEdge(source=s, target=t, edge_type="directed")
        for s, t in (edges or [])
    ]
    node_list = nodes if nodes is not None else ["t", "y", "x1", "x2"]
    return CausalDAG(
        nodes=node_list,
        edges=edge_list,
        discovery_method=method,
        treatment_variable="t",
        outcome_variable="y",
    )


# --- CAPABILITY -------------------------------------------------------------


class TestCapability:
    def test_name_is_causal_discovery(self):
        assert CAPABILITY.name == "causal_discovery"

    def test_needs_dataframe_path_and_data_profile(self):
        assert set(CAPABILITY.needs) == {"dataframe_path", "data_profile"}

    def test_does_not_need_treatment_or_outcome(self):
        # Discovery has a confounders-only fallback; missing primary pair
        # does not block the agent from emitting a DAG.
        assert "treatment_variable" not in CAPABILITY.needs
        assert "outcome_variable" not in CAPABILITY.needs

    def test_delivers_discovered_dag(self):
        assert "discovered_dag" in CAPABILITY.delivers

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            CAPABILITY.name = "other"

    def test_success_criteria_cover_each_flag_this_agent_raises(self):
        flags_with_criteria = {
            c.raises_flag for c in CAPABILITY.success_criteria if c.raises_flag
        }
        assert Flag.NEEDS_NOT_MET in flags_with_criteria
        assert Flag.LOW_STABILITY in flags_with_criteria
        assert Flag.CYCLE_DETECTED in flags_with_criteria


# --- preflight --------------------------------------------------------------


class TestPreflight:
    def test_refuses_when_dataframe_path_missing(self):
        state = _make_state(dataframe_path=None)
        brief = preflight(state)
        assert brief is not None
        assert brief.flags == [Flag.NEEDS_NOT_MET]
        assert "dataframe_path" in brief.headline

    def test_refuses_when_data_profile_missing(self):
        state = _make_state()
        state.data_profile = None
        brief = preflight(state)
        assert brief is not None
        assert brief.flags == [Flag.NEEDS_NOT_MET]
        assert "data_profile" in brief.headline

    def test_passes_when_needs_satisfied(self):
        state = _make_state()
        assert preflight(state) is None

    def test_passes_when_treatment_outcome_missing(self):
        state = _make_state()
        state.treatment_variable = None
        state.outcome_variable = None
        assert preflight(state) is None


# --- build_brief: status ----------------------------------------------------


class TestBuildBriefStatus:
    def test_failed_when_discovered_dag_is_none(self):
        state = _make_state(discovered_dag=None)
        brief = build_brief(state)
        assert brief.status == "failed"
        assert brief.artifact_keys == []
        assert "no dag" in brief.headline.lower()

    def test_done_when_discovered_dag_is_populated(self):
        dag = _dag(
            edges=[("t", "y"), ("x1", "y"), ("x1", "t")],
            method="PC (engine)",
        )
        state = _make_state(discovered_dag=dag)
        brief = build_brief(state)
        assert brief.status == "done"
        assert brief.artifact_keys == ["discovered_dag"]
        assert "PC" in brief.headline


# --- build_brief: low_stability --------------------------------------------


class TestBuildBriefLowStability:
    def test_raises_when_fewer_than_two_edges(self):
        dag = _dag(edges=[("t", "y")], method="PC (engine)")
        state = _make_state(discovered_dag=dag)
        brief = build_brief(state)
        assert Flag.LOW_STABILITY in brief.flags

    def test_raises_when_zero_edges(self):
        dag = _dag(edges=[], method="GES (engine)")
        state = _make_state(discovered_dag=dag)
        brief = build_brief(state)
        assert Flag.LOW_STABILITY in brief.flags

    def test_raises_when_discovery_method_is_fallback(self):
        # Even with enough edges, "Simple DAG (fallback)" means no
        # algorithm produced a usable graph; the orchestrator must know.
        dag = _dag(
            edges=[("x1", "t"), ("x1", "y"), ("t", "y")],
            method="Simple DAG (fallback)",
        )
        state = _make_state(discovered_dag=dag)
        brief = build_brief(state)
        assert Flag.LOW_STABILITY in brief.flags

    def test_does_not_raise_with_two_edges_from_real_algorithm(self):
        dag = _dag(
            edges=[("t", "y"), ("x1", "y")],
            method="PC (engine)",
        )
        state = _make_state(discovered_dag=dag)
        brief = build_brief(state)
        assert Flag.LOW_STABILITY not in brief.flags


# --- build_brief: cycle_detected -------------------------------------------


class TestBuildBriefCycleDetected:
    def test_raises_on_two_node_cycle(self):
        dag = _dag(
            edges=[("a", "b"), ("b", "a")],
            method="LINGAM (legacy)",
            nodes=["a", "b"],
        )
        state = _make_state(discovered_dag=dag)
        brief = build_brief(state)
        assert Flag.CYCLE_DETECTED in brief.flags

    def test_raises_on_three_node_cycle(self):
        dag = _dag(
            edges=[("a", "b"), ("b", "c"), ("c", "a")],
            method="LINGAM (legacy)",
            nodes=["a", "b", "c"],
        )
        state = _make_state(discovered_dag=dag)
        brief = build_brief(state)
        assert Flag.CYCLE_DETECTED in brief.flags

    def test_does_not_raise_on_acyclic_dag(self):
        dag = _dag(
            edges=[("a", "b"), ("a", "c"), ("b", "c")],
            method="PC (engine)",
            nodes=["a", "b", "c"],
        )
        state = _make_state(discovered_dag=dag)
        brief = build_brief(state)
        assert Flag.CYCLE_DETECTED not in brief.flags

    def test_undirected_self_pair_does_not_trigger_cycle(self):
        # Undirected edges represent uncertain orientation, not a cycle.
        dag = CausalDAG(
            nodes=["a", "b"],
            edges=[
                CausalEdge(source="a", target="b", edge_type="undirected"),
                CausalEdge(source="b", target="a", edge_type="undirected"),
            ],
            discovery_method="PC (engine)",
        )
        state = _make_state(discovered_dag=dag)
        brief = build_brief(state)
        assert Flag.CYCLE_DETECTED not in brief.flags


# --- build_brief: composite -------------------------------------------------


class TestBuildBriefAllFlags:
    def test_raises_low_stability_and_cycle_when_both_breached(self):
        # 1 edge + cycle (loop on a single edge target -> source).
        dag = CausalDAG(
            nodes=["a", "b"],
            edges=[
                CausalEdge(source="a", target="b", edge_type="directed"),
                CausalEdge(source="b", target="a", edge_type="directed"),
            ],
            discovery_method="LINGAM (legacy)",
        )
        state = _make_state(discovered_dag=dag)
        brief = build_brief(state)
        # 2 edges total so LOW_STABILITY does NOT fire on edge count,
        # but the LINGAM method is not a fallback either, so it should
        # only flag the cycle.
        assert Flag.CYCLE_DETECTED in brief.flags
        assert Flag.LOW_STABILITY not in brief.flags
        assert len(brief.raised_issues) == 1
