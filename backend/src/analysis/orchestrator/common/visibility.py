"""Turn an agent's sealed brief into the live findings and challenges the panel shows.

When control returns to the orchestrator the readiness gate already holds the
agent's AgentBrief (headline, status, flags, raised issues). This helper emits
that brief as SSE events so the UI surfaces, per agent and as the run proceeds,
what each agent found (agent_finding) and where it pushed back or hit a limit
(agent_challenge). A clean agent emits only a finding; a refusal, a failure, or
any quality flag also emits a challenge.

The event names are part of the wire contract pinned in
substrate/tests/test_events.py and consumed by the frontend.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .readiness import classify_brief

if TYPE_CHECKING:
    from src.analysis.agents.base import AnalysisState


def publish_brief_events(
    state: "AnalysisState", agent_name: str, verdict: str | None = None
) -> None:
    """Emit agent_finding (always) and agent_challenge (when not clean) for one agent.

    verdict is the classify_brief verdict when the caller already has it; when
    None it is recomputed from the brief so callers without it (the critique
    path) stay simple. No-op when the agent sealed no brief, mirroring the
    readiness gate's treatment of a missing brief.
    """
    brief = state.agent_briefs.get(agent_name)
    if brief is None:
        return

    if verdict is None:
        verdict, _ = classify_brief(brief)

    status = getattr(brief.status, "value", brief.status)
    flags = [f.value for f in brief.flags]

    state.push_sse_event(
        "agent_finding",
        {
            "agent_name": agent_name,
            "headline": brief.headline,
            "status": status,
            "flags": flags,
        },
    )

    if verdict != "ok":
        state.push_sse_event(
            "agent_challenge",
            {
                "agent_name": agent_name,
                "headline": brief.headline,
                "status": status,
                "flags": flags,
                "issues": list(brief.raised_issues),
            },
        )
