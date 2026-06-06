"""Contract surface for the confounder_discovery agent.

The agent runs an agentic loop (with a statistical fallback) that
identifies the variables most plausibly confounding the
treatment-outcome relationship. It writes a dict to
`state.confounder_discovery` with keys: ranked_confounders,
excluded_variables, adjustment_strategy, investigation_log.

Single flag from the closed enum: WEAK_CONFOUNDER_EVIDENCE when the
final ranked_confounders list is empty. This is the same flag
domain_knowledge raises when it cannot form a confounder hypothesis;
downstream specialists (ps_diagnostics, effect_estimator) read these
to decide whether to lean on data-driven discovery alone.
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

AGENT_NAME = "confounder_discovery"


CAPABILITY = AgentCapability(
    name=AGENT_NAME,
    answers=(
        "Which observed variables most plausibly confound the "
        "treatment-outcome relationship?"
    ),
    needs=("data_profile", "dataframe_path"),
    delivers=(
        "confounder_discovery",
        "agent_briefs[confounder_discovery]",
    ),
    refuses_when=(
        "data_profile is missing (profiler has not run)",
        "dataframe_path is missing (profiler has not loaded data)",
    ),
    success_criteria=(
        Criterion(
            id="confounder.brief.always_written",
            description=(
                "execute always writes a brief into "
                "state.agent_briefs['confounder_discovery']"
            ),
        ),
        Criterion(
            id="confounder.refusal.needs_missing",
            description=(
                "returns refusal brief with NEEDS_NOT_MET when "
                "data_profile or dataframe_path is missing"
            ),
            raises_flag=Flag.NEEDS_NOT_MET,
        ),
        Criterion(
            id="confounder.flag.weak_evidence",
            description=(
                "raises WEAK_CONFOUNDER_EVIDENCE when "
                "ranked_confounders is empty"
            ),
            raises_flag=Flag.WEAK_CONFOUNDER_EVIDENCE,
        ),
    ),
)


def preflight(state: AnalysisState) -> AgentBrief | None:
    """Refuse when the profiler hasn't produced the inputs we need."""
    if state.data_profile is None:
        return refusal_brief(
            agent=AGENT_NAME,
            flag=Flag.NEEDS_NOT_MET,
            headline="data_profile missing; profiler has not run",
            issues=["state.data_profile is None"],
        )
    if state.dataframe_path is None:
        return refusal_brief(
            agent=AGENT_NAME,
            flag=Flag.NEEDS_NOT_MET,
            headline="dataframe_path missing; profiler has not loaded data",
            issues=["state.dataframe_path is None"],
        )
    return None


def build_brief(state: AnalysisState) -> AgentBrief:
    """Shape the brief from state.confounder_discovery."""
    cd = state.confounder_discovery
    if not cd:
        return AgentBrief(
            agent=AGENT_NAME,
            status="failed",
            headline="confounder discovery produced no result",
            artifact_keys=[],
        )

    ranked = cd.get("ranked_confounders") or []
    excluded = cd.get("excluded_variables") or []

    flags: list[Flag] = []
    issues: list[str] = []

    if not ranked:
        flags.append(Flag.WEAK_CONFOUNDER_EVIDENCE)
        issues.append(
            "no confounders identified; downstream methods must rely on "
            "the data profiler's potential_confounders or DAG adjustment set"
        )

    headline = (
        f"{len(ranked)} confounder(s) ranked, {len(excluded)} excluded"
    )

    return AgentBrief(
        agent=AGENT_NAME,
        status="done",
        headline=headline,
        flags=flags,
        raised_issues=issues,
        artifact_keys=["confounder_discovery"],
    )
