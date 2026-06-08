"""Tests for publish_brief_events (orchestrator/common/visibility.py).

The helper turns a sealed AgentBrief into the live SSE events the panel shows:
a finding for every agent, plus a challenge when the brief is not clean. These
cases pin which events fire and that the challenge carries the flags and issues
a human needs to see.
"""
from __future__ import annotations

from src.analysis.agents import AnalysisState, DatasetInfo
from src.analysis.orchestrator.common import publish_brief_events
from src.domain.briefs import AgentBrief, Flag


def _state() -> AnalysisState:
    return AnalysisState(
        job_id="vis-test",
        dataset_info=DatasetInfo(url="test://x", name="x"),
    )


def _brief(status: str, flags=None, issues=None) -> AgentBrief:
    return AgentBrief(
        agent="dag_expert",
        status=status,
        headline="2 nodes, 1 edge, adjustment set of 0",
        flags=flags or [],
        raised_issues=issues or [],
    )


def _events(state: AnalysisState, event_type: str) -> list[dict]:
    return [e for e in state.sse_events if e["event_type"] == event_type]


def test_clean_brief_emits_only_a_finding():
    state = _state()
    state.agent_briefs["dag_expert"] = _brief("done")

    publish_brief_events(state, "dag_expert")

    findings = _events(state, "agent_finding")
    assert len(findings) == 1
    assert _events(state, "agent_challenge") == []
    data = findings[0]["data"]
    assert data["agent_name"] == "dag_expert"
    assert data["status"] == "done"
    assert data["flags"] == []
    assert data["headline"]


def test_soft_flag_emits_finding_and_challenge_with_flags_and_issues():
    state = _state()
    state.agent_briefs["dag_expert"] = _brief(
        "done", [Flag.NO_ADJUSTMENT_SET], ["no valid adjustment set found"]
    )

    publish_brief_events(state, "dag_expert")

    assert len(_events(state, "agent_finding")) == 1
    challenges = _events(state, "agent_challenge")
    assert len(challenges) == 1
    cdata = challenges[0]["data"]
    assert "no_adjustment_set" in cdata["flags"]
    assert cdata["issues"] == ["no valid adjustment set found"]


def test_refused_brief_emits_a_challenge():
    state = _state()
    state.agent_briefs["dag_expert"] = _brief("refused")

    publish_brief_events(state, "dag_expert")

    assert len(_events(state, "agent_finding")) == 1
    challenges = _events(state, "agent_challenge")
    assert len(challenges) == 1
    assert challenges[0]["data"]["status"] == "refused"


def test_missing_brief_emits_nothing():
    state = _state()

    publish_brief_events(state, "dag_expert")

    assert state.sse_events == []


def test_explicit_verdict_is_honored_over_recomputing():
    # The dispatch path passes the verdict it already computed; an explicit
    # "ok" suppresses the challenge even when the brief carries a flag, proving
    # the helper trusts the caller's verdict rather than always reclassifying.
    state = _state()
    state.agent_briefs["dag_expert"] = _brief("done", [Flag.NO_ADJUSTMENT_SET])

    publish_brief_events(state, "dag_expert", verdict="ok")

    assert len(_events(state, "agent_finding")) == 1
    assert _events(state, "agent_challenge") == []
