"""Slice 1 wiring: the orchestrator halts on a fatal brief, proceeds on a clean one.

These drive the real dispatch path (not classify_brief in isolation) so they pin
the wiring: a specialist that seals a cyclic-DAG brief stops the run with a reason
instead of letting the cycle flow into estimation; a refusal stops it too; a soft
flag is surfaced but does not stop; a clean brief is unchanged. The parallel case
also pins that a branch's brief now reaches the main state, which AGENT_WRITES
does not carry. End-to-end halting through the full run is covered in
test_spine_runner.py.
"""
from __future__ import annotations

import pytest

from src.analysis.agents.base.state import AnalysisState, DatasetInfo, JobStatus
from src.analysis.orchestrator.standard.agent import StandardOrchestrator
from src.domain.briefs import AgentBrief, Flag


class _StubSpecialist:
    """Minimal specialist: seals a given brief into state and returns it."""

    REQUIRED_STATE_FIELDS: tuple[str, ...] = ()

    def __init__(self, name: str, brief: AgentBrief | None) -> None:
        self._name = name
        self._brief = brief

    async def execute_with_tracing(self, state: AnalysisState) -> AnalysisState:
        if self._brief is not None:
            state.agent_briefs[self._name] = self._brief
        return state


def _state() -> AnalysisState:
    # No data_profile, so the data-review gate predicate does not fire and the
    # dispatch path is reached directly.
    return AnalysisState(job_id="job-1", dataset_info=DatasetInfo(url="kaggle.com/x"))


def _brief(agent: str, status: str, flags: list[Flag] | None = None) -> AgentBrief:
    return AgentBrief(
        agent=agent, status=status, headline=f"{agent} result", flags=flags or []
    )


def _args(name: str) -> dict:
    return {"agent_name": name, "task_description": "do it", "reasoning": "because"}


def _register(orch: StandardOrchestrator, name: str, brief: AgentBrief | None) -> None:
    orch.register_specialist(name, _StubSpecialist(name, brief))


@pytest.mark.asyncio
async def test_single_dispatch_halts_on_a_cyclic_dag_brief():
    orch = StandardOrchestrator()
    _register(orch, "dag_expert", _brief("dag_expert", "done", [Flag.CYCLE_DETECTED]))

    result = await orch._dispatch_to_agent(_state(), _args("dag_expert"))

    assert result.status == JobStatus.FAILED
    assert "cycle_detected" in (result.error_message or "")
    completed = [e for e in result.sse_events if e["event_type"] == "agent_completed"]
    assert completed and completed[-1]["data"]["success"] is False


@pytest.mark.asyncio
async def test_single_dispatch_halts_on_a_refusal_brief():
    orch = StandardOrchestrator()
    _register(orch, "dag_expert", _brief("dag_expert", "refused", [Flag.NEEDS_NOT_MET]))

    result = await orch._dispatch_to_agent(_state(), _args("dag_expert"))

    assert result.status == JobStatus.FAILED
    assert "refused" in (result.error_message or "")


@pytest.mark.asyncio
async def test_single_dispatch_surfaces_a_soft_flag_without_stopping():
    orch = StandardOrchestrator()
    _register(orch, "dag_expert", _brief("dag_expert", "done", [Flag.NO_ADJUSTMENT_SET]))

    result = await orch._dispatch_to_agent(_state(), _args("dag_expert"))

    assert result.status != JobStatus.FAILED
    assert any(d.decision_type == "readiness_flagged" for d in result.decisions)


@pytest.mark.asyncio
async def test_single_dispatch_is_unchanged_for_a_clean_brief():
    orch = StandardOrchestrator()
    _register(orch, "dag_expert", _brief("dag_expert", "done"))

    result = await orch._dispatch_to_agent(_state(), _args("dag_expert"))

    assert result.status != JobStatus.FAILED
    assert not any(d.decision_type == "readiness_flagged" for d in result.decisions)
    assert any(d.decision_type == "agent_dispatched" for d in result.decisions)


@pytest.mark.asyncio
async def test_single_dispatch_emits_a_finding_and_no_challenge_for_a_clean_brief():
    # Wiring: dispatch must publish the brief as a live finding; a clean brief
    # raises no challenge. Pins that the publish call exists and reads the right
    # agent's brief.
    orch = StandardOrchestrator()
    _register(orch, "dag_expert", _brief("dag_expert", "done"))

    result = await orch._dispatch_to_agent(_state(), _args("dag_expert"))

    findings = [e for e in result.sse_events if e["event_type"] == "agent_finding"]
    assert len(findings) == 1
    assert findings[0]["data"]["agent_name"] == "dag_expert"
    assert [e for e in result.sse_events if e["event_type"] == "agent_challenge"] == []


@pytest.mark.asyncio
async def test_single_dispatch_emits_a_challenge_for_a_flagged_brief():
    orch = StandardOrchestrator()
    _register(orch, "dag_expert", _brief("dag_expert", "done", [Flag.NO_ADJUSTMENT_SET]))

    result = await orch._dispatch_to_agent(_state(), _args("dag_expert"))

    challenges = [e for e in result.sse_events if e["event_type"] == "agent_challenge"]
    assert len(challenges) == 1
    assert challenges[0]["data"]["agent_name"] == "dag_expert"
    assert "no_adjustment_set" in challenges[0]["data"]["flags"]


@pytest.mark.asyncio
async def test_parallel_dispatch_emits_a_finding_per_branch():
    orch = StandardOrchestrator()
    _register(orch, "eda_agent", _brief("eda_agent", "done"))
    _register(orch, "causal_discovery", _brief("causal_discovery", "done"))

    args = {
        "agents": [
            {"agent_name": "eda_agent", "task_description": "x"},
            {"agent_name": "causal_discovery", "task_description": "y"},
        ],
        "reasoning": "independent learners",
    }
    result = await orch._dispatch_parallel(_state(), args)

    findings = [e for e in result.sse_events if e["event_type"] == "agent_finding"]
    assert {f["data"]["agent_name"] for f in findings} == {"eda_agent", "causal_discovery"}


@pytest.mark.asyncio
async def test_parallel_dispatch_merges_briefs_and_halts_on_a_fatal_one():
    orch = StandardOrchestrator()
    _register(orch, "eda_agent", _brief("eda_agent", "done"))
    _register(orch, "dag_expert", _brief("dag_expert", "done", [Flag.CYCLE_DETECTED]))

    args = {
        "agents": [
            {"agent_name": "eda_agent", "task_description": "x"},
            {"agent_name": "dag_expert", "task_description": "y"},
        ],
        "reasoning": "independent learners",
    }
    result = await orch._dispatch_parallel(_state(), args)

    # Both briefs reached the main state (AGENT_WRITES does not carry agent_briefs).
    assert "eda_agent" in result.agent_briefs
    assert "dag_expert" in result.agent_briefs
    # The cyclic brief stopped the run, after both branches merged.
    assert result.status == JobStatus.FAILED
    assert "cycle_detected" in (result.error_message or "")
