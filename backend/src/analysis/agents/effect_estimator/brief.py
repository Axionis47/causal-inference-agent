"""Contract surface for the effect_estimator.

The effect_estimator runs one or more causal estimation methods and
writes `state.treatment_effects: list[TreatmentEffectResult]`. The
brief reads from that list and derives three flags:

- METHOD_UNSTABLE: estimates disagree across methods (CV > 0.5),
  matching compare_estimates' own INCONSISTENT label.
- ATE_OUTLIER: at least one method's estimate sits more than 3 MAD
  from the median (matches the outlier check in compare_estimates).
- WEAK_INSTRUMENT: any IV result reports a small first-stage F (< 10)
  or the engine's `weak_instrument_severe` diagnostic is True.

Preflight refuses with NEEDS_NOT_MET when the profiler hasn't loaded
data; the agent's existing all-methods-failed mark_failed path is
preserved (it produces a status=failed brief in the finally block).
"""
from __future__ import annotations

import numpy as np

from src.analysis.agents.base import AnalysisState, TreatmentEffectResult
from src.domain.briefs import (
    AgentBrief,
    AgentCapability,
    Criterion,
    Flag,
    refusal_brief,
)

AGENT_NAME = "effect_estimator"

METHOD_UNSTABLE_CV_MAX = 0.5
ATE_OUTLIER_MAD_MULTIPLE = 3.0
WEAK_INSTRUMENT_F_MIN = 10.0


CAPABILITY = AgentCapability(
    name=AGENT_NAME,
    answers=(
        "What is the treatment effect, and how do estimates compare "
        "across appropriate causal-inference methods?"
    ),
    needs=("data_profile", "dataframe_path"),
    delivers=(
        "treatment_effects",
        "analyzed_pairs",
        "agent_briefs[effect_estimator]",
    ),
    refuses_when=(
        "data_profile is missing (profiler has not run)",
        "dataframe_path is missing (profiler has not loaded data)",
    ),
    success_criteria=(
        Criterion(
            id="estimator.brief.always_written",
            description=(
                "execute always writes a brief into "
                "state.agent_briefs['effect_estimator']"
            ),
        ),
        Criterion(
            id="estimator.refusal.needs_missing",
            description=(
                "returns refusal brief with NEEDS_NOT_MET when "
                "data_profile or dataframe_path is missing"
            ),
            raises_flag=Flag.NEEDS_NOT_MET,
        ),
        Criterion(
            id="estimator.flag.method_unstable",
            description=(
                "raises METHOD_UNSTABLE when the coefficient of "
                "variation across method estimates exceeds 0.5"
            ),
            raises_flag=Flag.METHOD_UNSTABLE,
        ),
        Criterion(
            id="estimator.flag.ate_outlier",
            description=(
                "raises ATE_OUTLIER when at least one estimate is "
                "more than 3 MAD from the median"
            ),
            raises_flag=Flag.ATE_OUTLIER,
        ),
        Criterion(
            id="estimator.flag.weak_instrument",
            description=(
                "raises WEAK_INSTRUMENT when an IV result reports "
                "first-stage F < 10 or weak_instrument_severe"
            ),
            raises_flag=Flag.WEAK_INSTRUMENT,
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
    """Shape the brief from state.treatment_effects."""
    effects = state.treatment_effects
    if not effects:
        return AgentBrief(
            agent=AGENT_NAME,
            status="failed",
            headline="estimator produced no treatment effects",
            artifact_keys=[],
        )

    flags: list[Flag] = []
    issues: list[str] = []

    cv = _coefficient_of_variation(effects)
    if cv is not None and cv > METHOD_UNSTABLE_CV_MAX:
        flags.append(Flag.METHOD_UNSTABLE)
        issues.append(
            f"cross-method CV is {cv:.2f} (> {METHOD_UNSTABLE_CV_MAX:.1f}); "
            "estimates disagree"
        )

    outliers = _mad_outliers(effects, ATE_OUTLIER_MAD_MULTIPLE)
    if outliers:
        flags.append(Flag.ATE_OUTLIER)
        issues.append(
            f"{len(outliers)} method(s) >3 MAD from median: {', '.join(outliers[:3])}"
        )

    weak = _weak_iv_methods(effects)
    if weak:
        flags.append(Flag.WEAK_INSTRUMENT)
        issues.append(
            f"weak instrument(s) detected in: {', '.join(weak[:3])}"
        )

    methods = sorted({e.method for e in effects})
    headline = (
        f"{len(effects)} estimate(s) across {len(methods)} method(s)"
    )

    return AgentBrief(
        agent=AGENT_NAME,
        status="done",
        headline=headline,
        flags=flags,
        raised_issues=issues,
        artifact_keys=["treatment_effects"],
    )


def _coefficient_of_variation(
    effects: list[TreatmentEffectResult],
) -> float | None:
    """CV of estimates, or None when fewer than 2 estimates."""
    if len(effects) < 2:
        return None
    estimates = np.array([e.estimate for e in effects])
    mean = float(np.mean(estimates))
    std = float(np.std(estimates))
    return std / (abs(mean) + 1e-10)


def _mad_outliers(
    effects: list[TreatmentEffectResult],
    multiple: float,
) -> list[str]:
    """Method names whose estimate is > `multiple` MAD from the median."""
    if len(effects) < 3:
        return []
    estimates = np.array([e.estimate for e in effects])
    median = float(np.median(estimates))
    mad = float(np.median(np.abs(estimates - median)))
    if mad == 0:
        return []
    return [
        e.method for e in effects
        if abs(e.estimate - median) > multiple * mad
    ]


def _weak_iv_methods(
    effects: list[TreatmentEffectResult],
) -> list[str]:
    """Method names that fail the IV first-stage strength check.

    The estimator engine writes one of two diagnostics for IV-family
    methods: `first_stage_f_partial` (continuous F-statistic) and
    `weak_instrument_severe` (boolean from the engine's own gate).
    Either one firing triggers the flag.
    """
    weak: list[str] = []
    for e in effects:
        details = e.details or {}
        diagnostics = (e.diagnostics or {}) if hasattr(e, "diagnostics") else {}
        merged = {**diagnostics, **details}

        if merged.get("weak_instrument_severe") is True:
            weak.append(e.method)
            continue

        f = merged.get("first_stage_f_partial")
        if isinstance(f, int | float) and f < WEAK_INSTRUMENT_F_MIN:
            weak.append(e.method)
    return weak
