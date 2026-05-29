"""Contract surface for the domain_knowledge agent.

Unlike ps_diagnostics, this agent aggregates everything into a typed
`DomainKnowledge` model at the end of `execute()`. So the brief reads
its signal from that typed model, not from metrics cached during the
ReAct loop. Same contract, different production pattern.

The completeness check uses the same `claim`-keyword logic as
`helpers.has_treatment_and_outcome_hypotheses`, so the brief never
disagrees with the agent's own `is_task_complete` signal.
"""
from __future__ import annotations

from src.analysis.agents.base.state import AnalysisState
from src.analysis.agents.domain_knowledge.output import DomainKnowledge, Hypothesis
from src.domain.briefs import (
    AgentBrief,
    AgentCapability,
    Criterion,
    Flag,
    refusal_brief,
)

AGENT_NAME = "domain_knowledge"

# Confidence levels that count as a "real" hypothesis, matching the
# bar used by helpers.has_treatment_and_outcome_hypotheses so the
# brief's notion of "done" lines up with the agent's completion gate.
CONFIDENT_LEVELS = ("medium", "high")


CAPABILITY = AgentCapability(
    name=AGENT_NAME,
    answers="What causal-role hypotheses do the dataset metadata support?",
    needs=("dataset_info",),
    delivers=(
        "domain_knowledge",
        "agent_briefs[domain_knowledge]",
    ),
    refuses_when=(
        "dataset_info carries no description, no column descriptions, "
        "no tags, and raw_metadata is empty (nothing to investigate)",
    ),
    success_criteria=(
        Criterion(
            id="dk.brief.always_written",
            description=(
                "execute always writes a brief into "
                "state.agent_briefs['domain_knowledge']"
            ),
        ),
        Criterion(
            id="dk.refusal.no_metadata",
            description=(
                "returns refusal brief with NEEDS_NOT_MET when every "
                "metadata source on the state is empty"
            ),
            raises_flag=Flag.NEEDS_NOT_MET,
        ),
        Criterion(
            id="dk.flag.weak_confounders",
            description=(
                "raises WEAK_CONFOUNDER_EVIDENCE when no confounder "
                "hypothesis was formed during the investigation"
            ),
            raises_flag=Flag.WEAK_CONFOUNDER_EVIDENCE,
        ),
        Criterion(
            id="dk.status.failed_when_incomplete",
            description=(
                "brief.status is 'failed' when the agent finished "
                "without both a treatment and an outcome hypothesis"
            ),
        ),
        Criterion(
            id="dk.status.done_when_complete",
            description=(
                "brief.status is 'done' when at least one treatment "
                "and one outcome hypothesis exist at medium-or-better "
                "confidence"
            ),
        ),
    ),
)


def preflight(state: AnalysisState) -> AgentBrief | None:
    """Refuse if there is no metadata at all to investigate.

    The agent reads description, column descriptions, tags, and the
    raw_metadata blob. If every one of these is empty there is nothing
    for the ReAct loop to do, and the orchestrator should reroute to
    the metadata-fetch stage rather than waste turns on this agent.
    """
    ds = state.dataset_info
    has_any_metadata = bool(
        ds.kaggle_description
        or ds.kaggle_column_descriptions
        or ds.kaggle_tags
        or ds.kaggle_domain
        or state.raw_metadata
    )
    if not has_any_metadata:
        return refusal_brief(
            agent=AGENT_NAME,
            flag=Flag.NEEDS_NOT_MET,
            headline="no metadata available to investigate",
            issues=[
                "dataset_info has no description, column descriptions, "
                "tags, or domain, and state.raw_metadata is empty",
            ],
        )
    return None


def build_brief(state: AnalysisState) -> AgentBrief:
    """Read the typed DomainKnowledge from state and shape the brief.

    Counts are derived from hypothesis claims using keyword matching,
    same as the agent's own completion check. `state.domain_knowledge`
    is None only if the ReAct loop crashed before the post-loop write
    in execute(); that case is reported as failed.
    """
    dk = state.domain_knowledge
    if dk is None:
        return AgentBrief(
            agent=AGENT_NAME,
            status="failed",
            headline="investigation produced no domain knowledge result",
            artifact_keys=[],
        )

    n_treatment = _count_confident(dk.hypotheses, "treatment")
    n_outcome = _count_confident(dk.hypotheses, "outcome")
    n_confounder = _count_confident(dk.hypotheses, "confound")
    n_uncertainties = len(dk.uncertainties)

    flags: list[Flag] = []
    issues: list[str] = []
    if n_confounder == 0:
        flags.append(Flag.WEAK_CONFOUNDER_EVIDENCE)
        issues.append(
            "no confounder hypothesis formed from metadata; "
            "downstream specialists must rely on data-driven discovery"
        )

    complete = n_treatment > 0 and n_outcome > 0
    status = "done" if complete else "failed"
    headline = _headline(n_treatment, n_outcome, n_confounder, n_uncertainties)
    artifact_keys = ["domain_knowledge"]

    return AgentBrief(
        agent=AGENT_NAME,
        status=status,
        headline=headline,
        flags=flags,
        raised_issues=issues,
        artifact_keys=artifact_keys,
    )


def _count_confident(hypotheses: list[Hypothesis], keyword: str) -> int:
    """Count live hypotheses whose claim mentions `keyword` at medium+ confidence.

    Lifted from the agent's completion check so the brief's view of
    "complete" lines up with the loop's exit condition.
    """
    return sum(
        1
        for h in hypotheses
        if keyword in h.claim.lower() and h.confidence in CONFIDENT_LEVELS
    )


def _headline(
    n_treatment: int,
    n_outcome: int,
    n_confounder: int,
    n_uncertainties: int,
) -> str:
    return (
        f"{n_treatment} treatment, {n_outcome} outcome, "
        f"{n_confounder} confounder hypotheses; "
        f"{n_uncertainties} uncertainties flagged"
    )
