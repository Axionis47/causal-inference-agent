"""Readiness gate: read an agent's sealed brief to decide if the run may advance.

Every specialist writes a typed AgentBrief into state.agent_briefs[name] on each
dispatch (status plus flags from the closed Flag enum). Until now the orchestrator
wrote those briefs but never read them, so a refusal or a cyclic DAG flowed
downstream and crashed a later agent instead of stopping here with a reason.

classify_brief reads one brief and returns a verdict:

    halt  the agent refused or failed, or raised a flag that makes downstream
          work unsafe (a cyclic DAG breaks identification). The run stops with
          the reason.
    soft  the agent finished but flagged a quality concern that has a documented
          fallback (no adjustment set, lost rows, imbalance, weak overlap). The
          run proceeds; the concern is surfaced.
    ok    clean, or no brief was sealed for this agent.

A missing brief is ok on purpose: not every dispatch path seals one, and this
gate must leave a clean run unchanged.
"""
from __future__ import annotations

from typing import Literal

from src.domain.briefs import AgentBrief, Flag

Verdict = Literal["halt", "soft", "ok"]

# Flags that make downstream analysis unsafe no matter the status. A directed
# cycle in the DAG means the effect is not identified, so estimation must not
# run on it. Other flags carry a documented fallback and are surfaced, not fatal.
_HALTING_FLAGS: frozenset[Flag] = frozenset({Flag.CYCLE_DETECTED})


def classify_brief(brief: AgentBrief | None) -> tuple[Verdict, str]:
    """Return (verdict, reason) for one agent's brief.

    reason is a short human-readable string for the failure (halt) or the
    surfaced concern (soft), and empty when ok. The string is safe to put in a
    job failure message or an SSE payload.
    """
    if brief is None:
        return "ok", ""

    if brief.status in ("refused", "failed"):
        return "halt", f"{brief.agent} {brief.status}: {brief.headline}"

    halting = [f for f in brief.flags if f in _HALTING_FLAGS]
    if halting:
        names = ", ".join(f.value for f in halting)
        return "halt", f"{brief.agent} raised {names}: {brief.headline}"

    if brief.flags:
        names = ", ".join(f.value for f in brief.flags)
        return "soft", f"{brief.agent} flagged {names}"

    return "ok", ""
