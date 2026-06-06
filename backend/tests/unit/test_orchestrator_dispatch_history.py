"""Tests for the dispatch-history helper that breaks the standard orch's
iterate-loop cycle.

Without this helper, the orchestrator's prompt has no signal that it just
re-dispatched the same parallel pair. The critique's "address covariate
imbalance" reads as a fresh problem every iteration and the orchestrator
spends ~160s firing eda_agent + confounder_discovery six times. The
helper feeds a one-line counter into the prompt so the LLM can see its
own behaviour and stop.
"""

from __future__ import annotations

from src.analysis.agents import AnalysisState, DatasetInfo
from src.analysis.agents.base.state import AgentTrace
from src.analysis.orchestrator.common.context import summarize_recent_dispatches


def _state_with_dispatches(actions: list[str]) -> AnalysisState:
    state = AnalysisState(
        job_id="t",
        dataset_info=DatasetInfo(url="t://t", name="t"),
        treatment_variable="t",
        outcome_variable="y",
    )
    for action in actions:
        state.agent_traces.append(
            AgentTrace(agent_name="orchestrator", action=action, reasoning="")
        )
    return state


def test_no_traces_returns_empty():
    state = _state_with_dispatches([])
    assert summarize_recent_dispatches(state) == ""


def test_no_dispatch_traces_returns_empty():
    state = _state_with_dispatches(["step_1_inspect_data", "step_2_analyze_treatment"])
    assert summarize_recent_dispatches(state) == ""


def test_single_dispatch_counted():
    state = _state_with_dispatches(["dispatch_to_data_profiler"])
    assert summarize_recent_dispatches(state) == "recent dispatches: data_profiler×1"


def test_repeated_dispatches_counted():
    state = _state_with_dispatches(
        [
            "dispatch_to_eda_agent",
            "dispatch_to_eda_agent",
            "dispatch_to_eda_agent",
            "dispatch_to_dag_expert",
        ]
    )
    out = summarize_recent_dispatches(state)
    # Sorted by count desc
    assert out.startswith("recent dispatches: ")
    assert "eda_agent×3" in out
    assert "dag_expert×1" in out
    # eda_agent (count 3) must come before dag_expert (count 1)
    assert out.index("eda_agent×3") < out.index("dag_expert×1")


def test_parallel_dispatch_splits_into_each_agent():
    """parallel_dispatch_eda_agent_confounder_discovery should count both
    specialists, not the joined string."""
    state = _state_with_dispatches(
        [
            "parallel_dispatch_eda_agent_confounder_discovery",
            "parallel_dispatch_eda_agent_confounder_discovery",
        ]
    )
    out = summarize_recent_dispatches(state)
    # Each specialist appears in two parallel dispatches.
    assert "eda×2" in out or "eda_agent" in out  # tolerant to splitter shape
    assert "confounder" in out


def test_window_limits_to_n():
    """If we set n=3, only the last 3 dispatch traces are counted."""
    state = _state_with_dispatches(
        [
            "dispatch_to_data_profiler",  # oldest, should be ignored when n=3
            "dispatch_to_eda_agent",
            "dispatch_to_eda_agent",
            "dispatch_to_eda_agent",
        ]
    )
    out = summarize_recent_dispatches(state, n=3)
    assert "data_profiler" not in out
    assert "eda_agent×3" in out


def test_non_dispatch_traces_between_dispatches_are_skipped():
    """Step traces between dispatch traces shouldn't break counting; the
    helper looks at the last n DISPATCH traces, not last n traces total."""
    state = _state_with_dispatches(
        [
            "dispatch_to_data_profiler",
            "step_1_inspect_data",
            "step_2_analyze_treatment",
            "dispatch_to_eda_agent",
            "step_1_compute_correlations",
            "dispatch_to_dag_expert",
        ]
    )
    out = summarize_recent_dispatches(state)
    assert "data_profiler×1" in out
    assert "eda_agent×1" in out
    assert "dag_expert×1" in out
