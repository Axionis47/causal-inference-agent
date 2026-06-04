"""Contract surface for the dataset_inspector agent.

Mirrors the sealed-brief pattern used by ps_diagnostics, eda,
domain_knowledge, and causal_discovery. The inspector is the first
agent that runs once Kaggle download has produced a file list; it
refuses if no data files are present (the orchestrator should reroute
to the loader), otherwise it always lands a brief.

`build_brief` reads what `agent.execute` has already written to state
(`file_profiles`, `data_profile`, the `file_selected` decision) and
shapes it into the headline + artifact pointers. Flag derivation is
deterministic: NEEDS_NOT_MET when no data files exist; no other flags
are raised today (the inspector does not assess causal-quality issues,
only structural ones).
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

AGENT_NAME = "dataset_inspector"


CAPABILITY = AgentCapability(
    name=AGENT_NAME,
    answers="Which file in the Kaggle bundle is the right one to analyze?",
    needs=("dataset_info",),
    delivers=(
        "file_profiles",
        "data_profile",
        "dataframe_path",
        "agent_briefs[dataset_inspector]",
    ),
    refuses_when=(
        "dataset_info.files contains no csv or parquet candidates",
    ),
    success_criteria=(
        Criterion(
            id="inspector.brief.always_written",
            description=(
                "execute always writes a brief into "
                "state.agent_briefs['dataset_inspector']"
            ),
        ),
        Criterion(
            id="inspector.refusal.no_data_files",
            description=(
                "returns refusal brief with NEEDS_NOT_MET when no csv or "
                "parquet files exist in dataset_info.files"
            ),
            raises_flag=Flag.NEEDS_NOT_MET,
        ),
        Criterion(
            id="inspector.profiles_written",
            description=(
                "file_profiles has one entry per candidate data file "
                "after execute returns"
            ),
        ),
        Criterion(
            id="inspector.winner_copied",
            description=(
                "data_profile is set to the winner's profile and "
                "dataframe_path points at an existing file"
            ),
        ),
        Criterion(
            id="inspector.decision_recorded",
            description=(
                "state.decisions gains a 'file_selected' entry with the "
                "winner and a one-line reason"
            ),
        ),
    ),
)


def _candidate_files(state: AnalysisState) -> list[str]:
    """Names of files in dataset_info.files that are real data candidates.

    csv / parquet only; skip README / json metadata / images. Defensive
    against missing format strings.
    """
    valid = {"csv", "parquet"}
    return [
        f.name for f in state.dataset_info.files
        if (f.format or "").lower() in valid
    ]


def preflight(state: AnalysisState) -> AgentBrief | None:
    """Return a refusal brief if no candidate data files exist."""
    if not _candidate_files(state):
        return refusal_brief(
            agent=AGENT_NAME,
            flag=Flag.NEEDS_NOT_MET,
            headline="no csv or parquet files in dataset_info.files",
            issues=[
                f"dataset_info.files = {[f.name for f in state.dataset_info.files] or '[]'}"
            ],
        )
    return None


def build_brief(state: AnalysisState) -> AgentBrief:
    """Shape from file_profiles + the file_selected decision.

    Called from agent.execute after the scorer ran. Reads the winner out
    of state.dataset_info.files (the entry with `used=True`) and the
    rationale out of the most recent file_selected decision.
    """
    profiles = state.file_profiles
    n = len(profiles)

    winner_name = next(
        (f.name for f in state.dataset_info.files if f.used),
        None,
    )

    if winner_name is None or winner_name not in profiles:
        return AgentBrief(
            agent=AGENT_NAME,
            status="failed",
            headline=f"profiled {n} files but no winner was selected",
            artifact_keys=["file_profiles"] if profiles else [],
        )

    winner = profiles[winner_name]
    headline = (
        f"profiled {n} file{'s' if n != 1 else ''}; picked {winner_name} "
        f"({winner.n_samples} rows, "
        f"{len(winner.treatment_candidates)} treatment, "
        f"{len(winner.outcome_candidates)} outcome)"
    )
    return AgentBrief(
        agent=AGENT_NAME,
        status="done",
        headline=headline[:200],
        artifact_keys=["file_profiles", "data_profile", "dataframe_path"],
    )
