"""Contract tests for the dag_expert brief."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis.agents.base import CausalDAG, CausalEdge
from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.dag_expert.brief import CAPABILITY, build_brief, preflight
from src.domain.briefs import Flag


def _dag(
    *,
    nodes: list[str] | None = None,
    edges: list[tuple[str, str]] | None = None,
    method: str = "DAG Expert (fusion)",
    adjustment_set: list[str] | None = None,
    forbidden_edges: list[dict[str, str]] | None = None,
    variable_roles: dict[str, str] | None = None,
) -> CausalDAG:
    edge_list = [
        CausalEdge(source=s, target=t, edge_type="directed")
        for s, t in (edges or [])
    ]
    return CausalDAG(
        nodes=nodes or ["t", "y", "x"],
        edges=edge_list,
        discovery_method=method,
        treatment_variable="t",
        outcome_variable="y",
        adjustment_set=adjustment_set,
        forbidden_edges=forbidden_edges,
        variable_roles=variable_roles,
    )


def _make_state(
    *,
    discovered_dag: CausalDAG | None = None,
    refined_dag: CausalDAG | None = None,
) -> AnalysisState:
    state = AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://example.com/data"),
    )
    state.discovered_dag = discovered_dag if discovered_dag is not None else _dag()
    state.refined_dag = refined_dag
    return state


class TestCapability:
    def test_name(self):
        assert CAPABILITY.name == "dag_expert"

    def test_needs(self):
        assert set(CAPABILITY.needs) == {"dataset_info", "discovered_dag"}

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            CAPABILITY.name = "other"

    def test_success_criteria_cover_each_flag(self):
        flags_with_criteria = {
            c.raises_flag for c in CAPABILITY.success_criteria if c.raises_flag
        }
        assert Flag.NEEDS_NOT_MET in flags_with_criteria
        assert Flag.NO_ADJUSTMENT_SET in flags_with_criteria
        assert Flag.DAG_CONFLICT in flags_with_criteria
        assert Flag.COLLIDER_SUSPECTED in flags_with_criteria
        assert Flag.CYCLE_DETECTED in flags_with_criteria


class TestPreflight:
    def test_refuses_when_discovered_dag_missing(self):
        state = _make_state()
        state.discovered_dag = None
        brief = preflight(state)
        assert brief is not None
        assert brief.flags == [Flag.NEEDS_NOT_MET]

    def test_passes_when_discovery_present(self):
        assert preflight(_make_state()) is None


class TestBuildBriefStatus:
    def test_failed_when_no_refined_dag(self):
        state = _make_state(refined_dag=None)
        brief = build_brief(state)
        assert brief.status == "failed"
        assert brief.artifact_keys == []

    def test_done_when_refined_dag_present(self):
        state = _make_state(refined_dag=_dag(adjustment_set=["x"]))
        brief = build_brief(state)
        assert brief.status == "done"
        assert brief.artifact_keys == ["refined_dag"]


class TestBuildBriefFlags:
    def test_no_adjustment_set_when_empty(self):
        state = _make_state(refined_dag=_dag(adjustment_set=None))
        brief = build_brief(state)
        assert Flag.NO_ADJUSTMENT_SET in brief.flags

    def test_no_adjustment_set_not_raised_when_populated(self):
        state = _make_state(refined_dag=_dag(adjustment_set=["x", "z"]))
        brief = build_brief(state)
        assert Flag.NO_ADJUSTMENT_SET not in brief.flags

    def test_dag_conflict_when_forbidden_edges_present(self):
        state = _make_state(refined_dag=_dag(
            adjustment_set=["x"],
            forbidden_edges=[{"source": "y", "target": "t", "reason": "reverse causality"}],
        ))
        brief = build_brief(state)
        assert Flag.DAG_CONFLICT in brief.flags

    def test_collider_suspected_when_variable_role_present(self):
        state = _make_state(refined_dag=_dag(
            adjustment_set=["x"],
            variable_roles={"x": "confounder", "z": "potential_collider"},
        ))
        brief = build_brief(state)
        assert Flag.COLLIDER_SUSPECTED in brief.flags

    def test_no_collider_when_roles_clean(self):
        state = _make_state(refined_dag=_dag(
            adjustment_set=["x"],
            variable_roles={"x": "confounder"},
        ))
        brief = build_brief(state)
        assert Flag.COLLIDER_SUSPECTED not in brief.flags

    def test_cycle_detected(self):
        state = _make_state(refined_dag=_dag(
            edges=[("a", "b"), ("b", "a")],
            nodes=["a", "b"],
            adjustment_set=["a"],
        ))
        brief = build_brief(state)
        assert Flag.CYCLE_DETECTED in brief.flags

    def test_no_cycle_on_acyclic(self):
        state = _make_state(refined_dag=_dag(
            edges=[("t", "y"), ("x", "y")],
            adjustment_set=["x"],
        ))
        brief = build_brief(state)
        assert Flag.CYCLE_DETECTED not in brief.flags
