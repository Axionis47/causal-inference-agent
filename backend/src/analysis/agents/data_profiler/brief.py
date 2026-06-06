"""Contract surface for the data_profiler agent.

The profiler is the entry point of the analysis pipeline. It loads
the dataset, computes column types, missingness, candidate
treatment/outcome variables, and (if the treatment is multi-level
categorical) writes a `TreatmentEncoding` that downstream methods
honor deterministically.

Because the profiler is the entry point, preflight has no meaningful
refusal case — state initialization always carries a `dataset_info`,
and the agent's own internal "could not load" path produces a
status=failed brief via the finally block. The NEEDS_NOT_MET flag is
still on the success-criteria list as a placeholder for future
upstream tightening; for now `preflight` returns None.

Flags from the closed enum:
- HIGH_MISSINGNESS: any column has more than 50% missing values
- ENCODING_REQUIRED: state.treatment_encoding was written with a
  non-"none" strategy (multi-level categorical treatment needs
  collapsing / label encoding)
"""
from __future__ import annotations

from src.analysis.agents.base import AnalysisState
from src.domain.briefs import (
    AgentBrief,
    AgentCapability,
    Criterion,
    Flag,
)

AGENT_NAME = "data_profiler"

HIGH_MISSINGNESS_PCT = 0.5


CAPABILITY = AgentCapability(
    name=AGENT_NAME,
    answers=(
        "What are the dataset's column types, missingness, candidate "
        "treatment/outcome variables, and required treatment encoding?"
    ),
    needs=("dataset_info",),
    delivers=(
        "data_profile",
        "dataframe_path",
        "treatment_encoding",
        "agent_briefs[data_profiler]",
    ),
    refuses_when=(),
    success_criteria=(
        Criterion(
            id="profiler.brief.always_written",
            description=(
                "execute always writes a brief into "
                "state.agent_briefs['data_profiler']"
            ),
        ),
        Criterion(
            id="profiler.flag.high_missingness",
            description=(
                "raises HIGH_MISSINGNESS when any column has more than "
                "50 percent missing values"
            ),
            raises_flag=Flag.HIGH_MISSINGNESS,
        ),
        Criterion(
            id="profiler.flag.encoding_required",
            description=(
                "raises ENCODING_REQUIRED when state.treatment_encoding "
                "carries a non-'none' strategy"
            ),
            raises_flag=Flag.ENCODING_REQUIRED,
        ),
    ),
)


def preflight(state: AnalysisState) -> AgentBrief | None:
    """No refusal: profiler is the entry point.

    See module docstring for the reasoning. Internal load failures
    surface as status=failed in build_brief.
    """
    return None


def build_brief(state: AnalysisState) -> AgentBrief:
    """Shape the brief from state.data_profile."""
    profile = state.data_profile
    if profile is None:
        return AgentBrief(
            agent=AGENT_NAME,
            status="failed",
            headline="profiler produced no DataProfile (load failure)",
            artifact_keys=[],
        )

    flags: list[Flag] = []
    issues: list[str] = []

    high_missing = [
        col for col, count in profile.missing_values.items()
        if profile.n_samples > 0
        and (count / profile.n_samples) > HIGH_MISSINGNESS_PCT
    ]
    if high_missing:
        flags.append(Flag.HIGH_MISSINGNESS)
        issues.append(
            f"{len(high_missing)} column(s) with >50% missing: "
            f"{', '.join(high_missing[:3])}"
        )

    encoding = state.treatment_encoding
    if encoding is not None and encoding.strategy != "none":
        flags.append(Flag.ENCODING_REQUIRED)
        issues.append(
            f"treatment is {encoding.original_type}; "
            f"encoded via strategy='{encoding.strategy}'"
        )

    headline = (
        f"{profile.n_samples} samples, {profile.n_features} features; "
        f"{len(profile.treatment_candidates)} treatment + "
        f"{len(profile.outcome_candidates)} outcome candidate(s)"
    )

    artifact_keys: list[str] = ["data_profile"]
    if state.dataframe_path:
        artifact_keys.append("dataframe_path")
    if state.treatment_encoding is not None:
        artifact_keys.append("treatment_encoding")

    return AgentBrief(
        agent=AGENT_NAME,
        status="done",
        headline=headline,
        flags=flags,
        raised_issues=issues,
        artifact_keys=artifact_keys,
    )
