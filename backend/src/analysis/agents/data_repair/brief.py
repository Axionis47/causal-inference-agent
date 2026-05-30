"""Contract surface for the data_repair agent.

The repair agent loads the dataframe the profiler wrote and applies
targeted fixes (missing-value imputation, outlier handling,
collinearity-induced column drops). It writes a list of repair-action
dicts to state.data_repairs.

Flags from the closed enum:
- REPAIR_FAILED: the loop ran but recorded no repairs at all despite
  the profile flagging issues; orchestrator should suspect a load /
  permission issue before trusting downstream analyses.
- DATA_LOST: cumulative `rows_dropped` exceeds 5% of the original
  sample; downstream estimation runs on a meaningfully smaller
  sample.
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

AGENT_NAME = "data_repair"

DATA_LOST_PCT_THRESHOLD = 0.05


CAPABILITY = AgentCapability(
    name=AGENT_NAME,
    answers=(
        "Which data quality issues need fixing before estimation, and "
        "what repairs were applied?"
    ),
    needs=("data_profile", "dataframe_path"),
    delivers=(
        "data_repairs",
        "agent_briefs[data_repair]",
    ),
    refuses_when=(
        "data_profile is missing (profiler has not run)",
        "dataframe_path is missing (profiler has not loaded data)",
    ),
    success_criteria=(
        Criterion(
            id="repair.brief.always_written",
            description=(
                "execute always writes a brief into "
                "state.agent_briefs['data_repair']"
            ),
        ),
        Criterion(
            id="repair.refusal.needs_missing",
            description=(
                "returns refusal brief with NEEDS_NOT_MET when "
                "data_profile or dataframe_path is missing"
            ),
            raises_flag=Flag.NEEDS_NOT_MET,
        ),
        Criterion(
            id="repair.flag.repair_failed",
            description=(
                "raises REPAIR_FAILED when the agent recorded zero "
                "repair actions despite the profiler flagging issues"
            ),
            raises_flag=Flag.REPAIR_FAILED,
        ),
        Criterion(
            id="repair.flag.data_lost",
            description=(
                "raises DATA_LOST when cumulative rows_dropped exceeds "
                "5 percent of the original sample"
            ),
            raises_flag=Flag.DATA_LOST,
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
    """Shape the brief from state.data_repairs + the profile's missingness signal."""
    repairs = state.data_repairs
    if repairs is None:
        return AgentBrief(
            agent=AGENT_NAME,
            status="failed",
            headline="data_repair produced no record",
            artifact_keys=[],
        )

    flags: list[Flag] = []
    issues: list[str] = []

    profile = state.data_profile
    had_issues = bool(
        profile
        and (
            any(count > 0 for count in profile.missing_values.values())
            or (profile.n_samples or 0) == 0
        )
    )
    if had_issues and len(repairs) == 0:
        flags.append(Flag.REPAIR_FAILED)
        issues.append(
            "profiler reported missingness but data_repair recorded no actions"
        )

    rows_dropped_total = 0
    for r in repairs:
        if isinstance(r, dict):
            dropped = r.get("rows_dropped")
            if isinstance(dropped, int | float):
                rows_dropped_total += int(dropped)

    if profile and profile.n_samples > 0:
        pct = rows_dropped_total / profile.n_samples
        if pct > DATA_LOST_PCT_THRESHOLD:
            flags.append(Flag.DATA_LOST)
            issues.append(
                f"{rows_dropped_total} row(s) dropped ({pct:.1%} of original); "
                f"downstream estimation runs on a smaller sample"
            )

    headline = (
        f"{len(repairs)} repair action(s); "
        f"{rows_dropped_total} row(s) dropped"
    )

    return AgentBrief(
        agent=AGENT_NAME,
        status="done",
        headline=headline,
        flags=flags,
        raised_issues=issues,
        artifact_keys=["data_repairs"],
    )
