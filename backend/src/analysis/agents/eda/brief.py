"""Contract surface for the eda_agent.

The EDA agent runs a ReAct loop over distributional, balance, outlier,
correlation, and VIF tools, and aggregates everything into a typed
`EDAResult` on the state. Like domain_knowledge, the brief reads from
that typed model rather than from per-tool caches.

Flag thresholds line up with the EDA tools' own labels so the brief
never disagrees with the LLM's narrative:

- skewness |z| > 1 is what `analyze_variable` already calls
  "highly_skewed".
- max covariate SMD > 0.25 is the textbook "severe imbalance"
  threshold (0.1 is the "imbalanced" floor used by check_balance, but
  0.1 fires routinely in observational data and is too noisy for an
  orchestrator-level flag).
- `high_correlations` is already filtered to pairs with |r| > 0.7 by
  the EDA tool, so non-empty is enough.
"""
from __future__ import annotations

from src.analysis.agents.base.state import AnalysisState
from src.analysis.agents.eda.output import EDAResult
from src.domain.briefs import (
    AgentBrief,
    AgentCapability,
    Criterion,
    Flag,
    refusal_brief,
)

AGENT_NAME = "eda_agent"

OUTCOME_SKEW_MAX = 1.0
TC_IMBALANCE_SMD_MAX = 0.25


CAPABILITY = AgentCapability(
    name=AGENT_NAME,
    answers=(
        "What distributional, balance, correlation, and outlier issues "
        "does this dataset carry?"
    ),
    needs=(
        "dataframe_path",
        "data_profile",
    ),
    delivers=(
        "eda_result",
        "agent_briefs[eda_agent]",
    ),
    refuses_when=(
        "dataframe_path is missing (profiler has not loaded data)",
        "data_profile is missing (profiler has not run)",
    ),
    success_criteria=(
        Criterion(
            id="eda.brief.always_written",
            description=(
                "execute always writes a brief into "
                "state.agent_briefs['eda_agent']"
            ),
        ),
        Criterion(
            id="eda.refusal.needs_missing",
            description=(
                "returns refusal brief with NEEDS_NOT_MET when "
                "dataframe_path or data_profile is missing"
            ),
            raises_flag=Flag.NEEDS_NOT_MET,
        ),
        Criterion(
            id="eda.flag.outcome_skewed",
            description=(
                "raises OUTCOME_SKEWED when the outcome variable's "
                "skewness exceeds 1.0 in absolute value"
            ),
            raises_flag=Flag.OUTCOME_SKEWED,
        ),
        Criterion(
            id="eda.flag.tc_imbalance",
            description=(
                "raises TC_IMBALANCE when the max covariate SMD "
                "exceeds 0.25"
            ),
            raises_flag=Flag.TC_IMBALANCE,
        ),
        Criterion(
            id="eda.flag.suspect_correlation",
            description=(
                "raises SUSPECT_CORRELATION when at least one variable "
                "pair has |r| > 0.7"
            ),
            raises_flag=Flag.SUSPECT_CORRELATION,
        ),
    ),
)


def preflight(state: AnalysisState) -> AgentBrief | None:
    """Refuse when EDA's required state slots are absent.

    No treatment / outcome check: EDA can still produce useful
    distributional, correlation, and quality findings without a primary
    pair declared. The balance flag simply won't fire in that case.
    """
    if state.dataframe_path is None:
        return refusal_brief(
            agent=AGENT_NAME,
            flag=Flag.NEEDS_NOT_MET,
            headline="dataframe_path missing; profiler has not loaded data",
            issues=["state.dataframe_path is None"],
        )
    if state.data_profile is None:
        return refusal_brief(
            agent=AGENT_NAME,
            flag=Flag.NEEDS_NOT_MET,
            headline="data_profile missing; profiler has not run",
            issues=["state.data_profile is None"],
        )
    return None


def build_brief(state: AnalysisState) -> AgentBrief:
    """Shape the brief from the typed EDAResult.

    The EDA agent writes `state.eda_result` after the ReAct loop. If
    that field is still None (load failure, exception before the
    populate step), the brief is `status="failed"` so the orchestrator
    sees the agent ran but produced nothing usable.
    """
    eda = state.eda_result
    if eda is None:
        return AgentBrief(
            agent=AGENT_NAME,
            status="failed",
            headline="EDA produced no result (load or loop failure)",
            artifact_keys=[],
        )

    flags: list[Flag] = []
    issues: list[str] = []

    outcome_var = state.outcome_variable
    skewness = _outcome_skewness(eda, outcome_var)
    if skewness is not None and abs(skewness) > OUTCOME_SKEW_MAX:
        flags.append(Flag.OUTCOME_SKEWED)
        issues.append(
            f"outcome '{outcome_var}' skewness is {skewness:.2f} "
            f"(|.| > {OUTCOME_SKEW_MAX:.1f})"
        )

    max_smd = _max_covariate_smd(eda)
    if max_smd is not None and max_smd > TC_IMBALANCE_SMD_MAX:
        flags.append(Flag.TC_IMBALANCE)
        issues.append(
            f"max covariate SMD is {max_smd:.2f} "
            f"(> {TC_IMBALANCE_SMD_MAX:.2f} severe-imbalance threshold)"
        )

    n_high_corr = len(eda.high_correlations)
    if n_high_corr > 0:
        flags.append(Flag.SUSPECT_CORRELATION)
        issues.append(
            f"{n_high_corr} variable pair(s) with |r| > 0.7"
        )

    headline = (
        f"quality {eda.data_quality_score:.2f}, "
        f"{len(eda.data_quality_issues)} issue(s), "
        f"{n_high_corr} high-corr pair(s)"
    )

    return AgentBrief(
        agent=AGENT_NAME,
        status="done",
        headline=headline,
        flags=flags,
        raised_issues=issues,
        artifact_keys=["eda_result"],
    )


def _outcome_skewness(eda: EDAResult, outcome_var: str | None) -> float | None:
    """Pull outcome skewness out of distribution_stats safely.

    distribution_stats[col] is dict[str, Any] in EDAResult. The EDA tool
    stores skewness as a rounded float under the key "skewness".
    """
    if not outcome_var:
        return None
    stats = eda.distribution_stats.get(outcome_var)
    if not isinstance(stats, dict):
        return None
    value = stats.get("skewness")
    if isinstance(value, int | float):
        return float(value)
    return None


def _max_covariate_smd(eda: EDAResult) -> float | None:
    """Largest absolute SMD across the covariate_balance table.

    covariate_balance is dict[covariate] -> dict with "smd" key, written
    by tools/check_balance.py. Returns None when the balance step did
    not run (e.g., treatment variable was not set).
    """
    if not eda.covariate_balance:
        return None
    smds: list[float] = []
    for stats in eda.covariate_balance.values():
        if isinstance(stats, dict):
            value = stats.get("smd")
            if isinstance(value, int | float):
                smds.append(abs(float(value)))
    if not smds:
        return None
    return max(smds)
