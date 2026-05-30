"""Contract surface for the critique agent.

Critique is the final review stage. It synthesizes signals from
upstream agents (effect_estimator, sensitivity_analyst,
ps_diagnostics) into a single APPROVE / ITERATE / REJECT decision
and a per-dimension score. The brief reads from the latest
`state.critique_history[-1]: CritiqueFeedback`.

Flag philosophy: critique does **not** raise primary issue flags
from the closed enum. The orchestrator already routes off
`state.critique_history[-1].decision` (APPROVE iterates the pipeline
to notebook; ITERATE retries upstream; REJECT halts). Raising e.g.
NOT_ROBUST here would duplicate the signal sensitivity_analyst
already raised. Critique's role in the brief vocabulary is the
status + headline + issues list, not new flags.

Preflight refuses with NEEDS_NOT_MET when state.treatment_effects is
empty: there is nothing to critique. The earlier silent path
(critique runs, auto_finalize emits REJECT) is preserved when the
LLM execution exits cleanly with auto_finalize; the preflight catches
the *orchestrator* mistakenly dispatching critique before
effect_estimator has run.
"""
from __future__ import annotations

from src.analysis.agents.base.state import AnalysisState
from src.domain.briefs import (
    AgentBrief,
    AgentCapability,
    Criterion,
    Flag,
    refusal_brief,
)

AGENT_NAME = "critique"


CAPABILITY = AgentCapability(
    name=AGENT_NAME,
    answers=(
        "Is the causal analysis methodologically sound enough to ship, "
        "or should the pipeline iterate or reject?"
    ),
    needs=("treatment_effects",),
    delivers=(
        "critique_history",
        "agent_briefs[critique]",
    ),
    refuses_when=(
        "state.treatment_effects is empty (effect_estimator has not produced estimates)",
    ),
    success_criteria=(
        Criterion(
            id="critique.brief.always_written",
            description=(
                "execute always writes a brief into "
                "state.agent_briefs['critique']"
            ),
        ),
        Criterion(
            id="critique.refusal.needs_missing",
            description=(
                "returns refusal brief with NEEDS_NOT_MET when "
                "state.treatment_effects is empty"
            ),
            raises_flag=Flag.NEEDS_NOT_MET,
        ),
        Criterion(
            id="critique.status.reflects_decision",
            description=(
                "brief status is done when a feedback was appended; "
                "headline reports the decision (APPROVE / ITERATE / REJECT)"
            ),
        ),
    ),
)


def preflight(state: AnalysisState) -> AgentBrief | None:
    """Refuse when there are no estimates to critique."""
    if not state.treatment_effects:
        return refusal_brief(
            agent=AGENT_NAME,
            flag=Flag.NEEDS_NOT_MET,
            headline=(
                "no treatment_effects to critique; "
                "effect_estimator has not produced estimates"
            ),
            issues=["state.treatment_effects is empty"],
        )
    return None


def build_brief(state: AnalysisState) -> AgentBrief:
    """Shape the brief from the latest CritiqueFeedback in critique_history."""
    if not state.critique_history:
        return AgentBrief(
            agent=AGENT_NAME,
            status="failed",
            headline="critique produced no feedback (loop or fallback failure)",
            artifact_keys=[],
        )

    feedback = state.critique_history[-1]
    avg_score = (
        sum(feedback.scores.values()) / len(feedback.scores)
        if feedback.scores else 0.0
    )

    headline = (
        f"{feedback.decision.value} on iteration {feedback.iteration}; "
        f"avg score {avg_score:.1f}/5; "
        f"{len(feedback.issues)} issue(s)"
    )

    return AgentBrief(
        agent=AGENT_NAME,
        status="done",
        headline=headline,
        flags=[],
        raised_issues=[_truncate(issue, 280) for issue in feedback.issues],
        artifact_keys=["critique_history"],
    )


def _truncate(s: str, max_len: int) -> str:
    """raised_issues entries must be <= 280 chars per AgentBrief validation."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
