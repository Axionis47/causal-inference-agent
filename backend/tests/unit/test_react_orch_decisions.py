"""Tests that the ReAct orchestrator surfaces decisions in state.

The react orchestrator deep-copies state before each specialist runs and
merges back only declared WRITES_STATE_FIELDS. Until this commit,
`decisions` was not in any specialist's whitelist, so every
push_decision() inside a specialist was silently dropped at the
isolation boundary. The orchestrator also did not record its own
dispatch decisions. End result on a real run: state.decisions stayed
empty across hundreds of agent steps, the notebook's Methodology
Decisions section rendered the skip placeholder, and the
/jobs/{id}/results.decision_log endpoint returned [].

These tests pin the fix:
    - specialist-pushed decisions cross the merge boundary
    - the orchestrator records its own dispatch as an audit-trail entry
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.analysis.agents import AnalysisState, DatasetInfo
from src.analysis.orchestrator.react import ReActOrchestrator


def _make_state() -> AnalysisState:
    return AnalysisState(
        job_id="test-react-decisions",
        dataset_info=DatasetInfo(url="https://kaggle.com/test", name="test"),
        treatment_variable="t",
        outcome_variable="y",
    )


@pytest.mark.asyncio
async def test_specialist_decisions_survive_state_merge():
    """A push_decision call inside a specialist must end up in state.decisions
    after the orchestrator's deep-copy/merge dance, even though the specialist
    declares no writes covering the decisions field.
    """
    orch = ReActOrchestrator()

    async def fake_execute(state):
        # Specialist records an internal methodology choice.
        state.push_decision(
            agent="data_profiler",
            decision_type="treatment_selection",
            choice="treat",
            reason="Binary indicator with balanced classes; matches Lalonde convention",
        )
        return state

    profiler = AsyncMock()
    profiler.execute_with_tracing.side_effect = fake_execute
    profiler.WRITES_STATE_FIELDS = []  # explicitly nothing
    orch.register_specialist("data_profiler", profiler)

    state = _make_state()
    await orch._dispatch_agent(
        state,
        agent_name="data_profiler",
        reasoning="Need to confirm treatment binarisation before estimation",
    )

    # The specialist's decision crossed the merge boundary.
    profiler_decisions = [d for d in state.decisions if d.agent == "data_profiler"]
    assert len(profiler_decisions) == 1
    assert profiler_decisions[0].decision_type == "treatment_selection"
    assert profiler_decisions[0].choice == "treat"


@pytest.mark.asyncio
async def test_orchestrator_logs_its_own_dispatch_decision():
    """Every dispatch is itself a methodology choice and belongs in the
    audit trail next to specialist decisions."""
    orch = ReActOrchestrator()

    profiler = AsyncMock()
    profiler.execute_with_tracing.side_effect = lambda s: s
    profiler.WRITES_STATE_FIELDS = []
    orch.register_specialist("data_profiler", profiler)

    state = _make_state()
    await orch._dispatch_agent(
        state,
        agent_name="data_profiler",
        reasoning="Profile the data first so domain_knowledge has variable types",
    )

    dispatched = [
        d for d in state.decisions
        if d.agent == "orchestrator" and d.decision_type == "agent_dispatched"
    ]
    assert len(dispatched) == 1
    assert dispatched[0].choice == "data_profiler"
    # Reason carries the LLM's stated rationale (truncated), not just a stub.
    assert "domain_knowledge" in dispatched[0].reason
