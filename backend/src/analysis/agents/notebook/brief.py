"""Contract surface for the notebook_generator agent.

The notebook agent is the terminal stage: it renders a reproducible
Jupyter notebook reporting pipeline findings and verification cells.
Each section renders unconditionally; the per-section render functions
emit "Section skipped" placeholders when their upstream stage
produced nothing.

Flag philosophy: notebook_generator raises **no closed-enum issue
flags**. Like the critique agent, this stage synthesizes upstream
state into output rather than detecting new analysis problems.
Primary issue flags belong to the upstream agents that detected
them (effect_estimator, sensitivity_analyst, ps_diagnostics, etc.).
The brief here is purely a status + headline signal: did the
notebook write successfully?

Preflight refuses NEEDS_NOT_MET when state.data_profile is missing —
there is nothing meaningful to render without it.
"""
from __future__ import annotations

from src.analysis.agents.base import AnalysisState
from src.domain.briefs import (
    AgentBrief,
    AgentCapability,
    Criterion,
    Flag,
    refusal_brief,
)

AGENT_NAME = "notebook_generator"


CAPABILITY = AgentCapability(
    name=AGENT_NAME,
    answers=(
        "Render the analysis as a reproducible Jupyter notebook with "
        "per-agent sections and executable verification cells."
    ),
    needs=("data_profile",),
    delivers=(
        "notebook_path",
        "agent_briefs[notebook_generator]",
    ),
    refuses_when=(
        "data_profile is missing (profiler has not run)",
    ),
    success_criteria=(
        Criterion(
            id="notebook.brief.always_written",
            description=(
                "execute always writes a brief into "
                "state.agent_briefs['notebook_generator']"
            ),
        ),
        Criterion(
            id="notebook.refusal.needs_missing",
            description=(
                "returns refusal brief with NEEDS_NOT_MET when "
                "data_profile is missing"
            ),
            raises_flag=Flag.NEEDS_NOT_MET,
        ),
        Criterion(
            id="notebook.status.reflects_write",
            description=(
                "status is done when state.notebook_path is set; "
                "failed otherwise"
            ),
        ),
    ),
)


def preflight(state: AnalysisState) -> AgentBrief | None:
    """Refuse when there is no profile to render."""
    if state.data_profile is None:
        return refusal_brief(
            agent=AGENT_NAME,
            flag=Flag.NEEDS_NOT_MET,
            headline="data_profile missing; nothing meaningful to render",
            issues=["state.data_profile is None"],
        )
    return None


def build_brief(state: AnalysisState) -> AgentBrief:
    """Shape the brief from state.notebook_path."""
    if not state.notebook_path:
        return AgentBrief(
            agent=AGENT_NAME,
            status="failed",
            headline="notebook was not written (rendering or save failure)",
            artifact_keys=[],
        )

    headline = f"notebook written to {state.notebook_path}"
    if len(headline) > 200:
        headline = f"notebook written ({len(state.notebook_path)}-char path)"

    return AgentBrief(
        agent=AGENT_NAME,
        status="done",
        headline=headline,
        flags=[],
        raised_issues=[],
        artifact_keys=["notebook_path"],
    )
