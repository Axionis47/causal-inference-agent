"""Contract surface for the ps_diagnostics agent.

This module owns three things the orchestrator depends on:

- `CAPABILITY`: a frozen `AgentCapability` declaration. The orchestrator
  reads `name`, `answers`, `needs`, and `success_criteria` to assemble
  its capability menu and decide when this worker is unblocked.
- `preflight`: a pure function that returns a refusal `AgentBrief` if
  the state does not satisfy `CAPABILITY.needs`. Returns None when the
  agent is safe to run. Called at the top of `agent.execute()`.
- `build_brief`: a pure function that turns the metrics cached during
  the ReAct loop into a typed brief. Flag derivation is deterministic
  (threshold-based), so the orchestrator's view of the result never
  depends on the LLM's wording.

The ReAct loop in `agent.py` is untouched. This file holds the contract,
not the analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.analysis.agents.base.state import AnalysisState
from src.domain.briefs import (
    AgentBrief,
    AgentCapability,
    Criterion,
    Flag,
    refusal_brief,
)

AGENT_NAME = "ps_diagnostics"

# Thresholds used both for flag derivation and as the source of the
# numbers quoted in headline / raised_issues. Match the agent's existing
# prompt thresholds so the brief never disagrees with the LLM's framing.
OVERLAP_MIN_PCT = 90.0
SMD_WEIGHTED_MAX = 0.1
CALIBRATION_MAE_MAX = 0.1


CAPABILITY = AgentCapability(
    name=AGENT_NAME,
    answers="Are propensity scores well-overlapped, balanced, and calibrated?",
    needs=(
        "dataframe_path",
        "data_profile",
        "treatment_variable",
        "outcome_variable",
    ),
    delivers=(
        "ps_diagnostics",
        "agent_briefs[ps_diagnostics]",
    ),
    refuses_when=(
        "dataframe_path is missing (profiler has not loaded data)",
        "data_profile is missing (profiler has not run)",
        "treatment_variable or outcome_variable is unset",
    ),
    success_criteria=(
        Criterion(
            id="ps.brief.always_written",
            description=(
                "execute always writes a brief into "
                "state.agent_briefs['ps_diagnostics']"
            ),
        ),
        Criterion(
            id="ps.refusal.needs_missing",
            description=(
                "returns refusal brief with NEEDS_NOT_MET when "
                "dataframe_path or data_profile is missing"
            ),
            raises_flag=Flag.NEEDS_NOT_MET,
        ),
        Criterion(
            id="ps.refusal.precondition",
            description=(
                "returns refusal brief with PRECONDITION_FAILED when "
                "treatment_variable or outcome_variable is unset"
            ),
            raises_flag=Flag.PRECONDITION_FAILED,
        ),
        Criterion(
            id="ps.flag.poor_overlap",
            description="raises POOR_OVERLAP when common support is below 90%",
            raises_flag=Flag.POOR_OVERLAP,
        ),
        Criterion(
            id="ps.flag.smd_high",
            description="raises SMD_HIGH when max weighted SMD exceeds 0.1",
            raises_flag=Flag.SMD_HIGH,
        ),
        Criterion(
            id="ps.flag.calibration_off",
            description="raises CALIBRATION_OFF when calibration MAE exceeds 0.1",
            raises_flag=Flag.CALIBRATION_OFF,
        ),
    ),
)


@dataclass
class Metrics:
    """Cached during the ReAct loop, read at the end to build the brief.

    The agent's tools populate these as they compute. Anything that
    remains None at brief-build time means that tool was not called
    (or failed before it produced a value).
    """

    overlap_pct: float | None = None
    max_smd_weighted: float | None = None
    calibration_mae: float | None = None

    def any_captured(self) -> bool:
        return (
            self.overlap_pct is not None
            or self.max_smd_weighted is not None
            or self.calibration_mae is not None
        )


def preflight(state: AnalysisState) -> AgentBrief | None:
    """Return a refusal brief if the state cannot satisfy this agent's needs.

    Order matters: dataframe_path and data_profile failures are
    NEEDS_NOT_MET (the orchestrator should reroute to the profiler).
    Missing treatment/outcome is PRECONDITION_FAILED (the orchestrator
    should escalate; nothing else can fix it).
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
    if not state.treatment_variable:
        return refusal_brief(
            agent=AGENT_NAME,
            flag=Flag.PRECONDITION_FAILED,
            headline="treatment_variable is unset on state",
        )
    if not state.outcome_variable:
        return refusal_brief(
            agent=AGENT_NAME,
            flag=Flag.PRECONDITION_FAILED,
            headline="outcome_variable is unset on state",
        )
    return None


def build_brief(state: AnalysisState, metrics: Metrics) -> AgentBrief:
    """Turn cached metrics into a typed brief.

    Flag derivation is deterministic and threshold-based: the LLM does
    not get to invent or omit flags. If no metric was captured at all,
    the brief is `status="failed"` with an empty flag list, so the
    orchestrator can see the agent ran but produced nothing usable.
    """
    flags: list[Flag] = []
    issues: list[str] = []

    if metrics.overlap_pct is not None and metrics.overlap_pct < OVERLAP_MIN_PCT:
        flags.append(Flag.POOR_OVERLAP)
        issues.append(
            f"common support is {metrics.overlap_pct:.1f}% "
            f"(< {OVERLAP_MIN_PCT:.0f}% threshold)"
        )
    if (
        metrics.max_smd_weighted is not None
        and metrics.max_smd_weighted > SMD_WEIGHTED_MAX
    ):
        flags.append(Flag.SMD_HIGH)
        issues.append(
            f"max weighted SMD is {metrics.max_smd_weighted:.2f} "
            f"(> {SMD_WEIGHTED_MAX:.2f} threshold)"
        )
    if (
        metrics.calibration_mae is not None
        and metrics.calibration_mae > CALIBRATION_MAE_MAX
    ):
        flags.append(Flag.CALIBRATION_OFF)
        issues.append(
            f"calibration MAE is {metrics.calibration_mae:.2f} "
            f"(> {CALIBRATION_MAE_MAX:.2f} threshold)"
        )

    status = "done" if metrics.any_captured() else "failed"
    headline = _headline(metrics, status)
    artifact_keys = ["ps_diagnostics"] if state.ps_diagnostics else []

    return AgentBrief(
        agent=AGENT_NAME,
        status=status,
        headline=headline,
        flags=flags,
        raised_issues=issues,
        artifact_keys=artifact_keys,
    )


def _headline(metrics: Metrics, status: str) -> str:
    if status == "failed":
        return "no diagnostics captured (overlap, balance, calibration all absent)"
    parts: list[str] = []
    if metrics.overlap_pct is not None:
        parts.append(f"overlap {metrics.overlap_pct:.0f}%")
    if metrics.max_smd_weighted is not None:
        parts.append(f"max weighted SMD {metrics.max_smd_weighted:.2f}")
    if metrics.calibration_mae is not None:
        parts.append(f"calibration MAE {metrics.calibration_mae:.2f}")
    return ", ".join(parts)
