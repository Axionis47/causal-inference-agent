"""Ranked design candidates from question type + lane eligibility.

The primary candidate follows the question type's natural lane. When that
lane is disabled, the honest fallback is covariate adjustment
(observational) carrying a warning, never a silently switched design.
"""
from __future__ import annotations

from src.analysis_v2.spec import (
    CausalSpec,
    Confidence,
    DesignCandidate,
    EligibilityState,
    LaneEligibility,
    MethodLane,
    QuestionType,
    ToolEligibility,
)

PRIMARY_LANE: dict[QuestionType, MethodLane] = {
    QuestionType.SIMPLE_EFFECT: MethodLane.OBSERVATIONAL,
    QuestionType.BINARY_TREATMENT: MethodLane.OBSERVATIONAL,
    QuestionType.DOSE_RESPONSE: MethodLane.OBSERVATIONAL,
    QuestionType.MULTI_FACTOR: MethodLane.OBSERVATIONAL,
    QuestionType.INTERACTION: MethodLane.OBSERVATIONAL,
    QuestionType.MEDIATION: MethodLane.MEDIATION,
    QuestionType.DID: MethodLane.DID,
    QuestionType.RDD: MethodLane.RDD,
    QuestionType.IV: MethodLane.IV,
    QuestionType.TIME_SERIES_INTERVENTION: MethodLane.TIME_SERIES,
    QuestionType.SURVIVAL: MethodLane.SURVIVAL,
    QuestionType.BEFORE_AFTER: MethodLane.TIME_SERIES,
    QuestionType.HETEROGENEOUS_EFFECTS: MethodLane.OBSERVATIONAL,
    QuestionType.DRIVER_ANALYSIS: MethodLane.OBSERVATIONAL,
    QuestionType.NO_EFFECT: MethodLane.OBSERVATIONAL,
    # which-pathway questions are mediation questions; the observational
    # fallback still applies when no mediator is resolved
    QuestionType.MECHANISM_SEARCH: MethodLane.MEDIATION,
}

_LABELS: dict[MethodLane, str] = {
    MethodLane.OBSERVATIONAL: "covariate-adjusted observational comparison",
    MethodLane.MATCHING: "propensity matching / weighting on a binary treatment",
    MethodLane.DID: "difference-in-differences across groups and periods",
    MethodLane.RDD: "regression discontinuity at the assignment cutoff",
    MethodLane.IV: "two-stage least squares with the supplied instrument",
    MethodLane.TIME_SERIES: "interrupted time-series around the intervention",
    MethodLane.MEDIATION: "mediation decomposition (direct/indirect effects)",
    MethodLane.SURVIVAL: "time-to-event analysis with censoring",
}


def _roles(spec: CausalSpec) -> dict[str, str]:
    roles: dict[str, str] = {}
    pairs = {
        "outcome": spec.outcome, "treatment": spec.treatment,
        "mediator": spec.mediator, "instrument": spec.instrument,
        "running_variable": spec.running_variable, "time": spec.time_column,
        "group": spec.group_column, "event": spec.event_column,
        "duration": spec.duration_column,
    }
    for role, ref in pairs.items():
        if ref is not None and ref.resolved:
            roles[role] = ref.column
    return roles


def _candidate(spec: CausalSpec, entry: LaneEligibility, *, primary: bool) -> DesignCandidate:
    confident = entry.state == EligibilityState.ENABLED and primary
    confidence = (
        Confidence.HIGH if confident and spec.confidence == Confidence.HIGH
        else Confidence.MEDIUM if entry.state != EligibilityState.DISABLED
        else Confidence.LOW
    )
    warnings = []
    if entry.state == EligibilityState.DISABLED:
        warnings.append(f"lane disabled: {entry.reason}")
    return DesignCandidate(
        lane=entry.lane,
        design_label=_LABELS[entry.lane],
        confidence=confidence,
        rationale=entry.reason,
        required_fields=_roles(spec),
        missing_requirements=list(entry.missing_requirements),
        warnings=warnings,
    )


def build_candidates(spec: CausalSpec, eligibility: ToolEligibility) -> list[DesignCandidate]:
    """Primary lane first, then usable alternatives. Disabled lanes are
    never offered as candidates; they live in the eligibility table with
    their reasons."""
    primary_lane = PRIMARY_LANE[spec.question_type]
    out: list[DesignCandidate] = []

    primary = eligibility.for_lane(primary_lane)
    if primary is not None and primary.state != EligibilityState.DISABLED:
        out.append(_candidate(spec, primary, primary=True))

    if primary_lane != MethodLane.OBSERVATIONAL:
        fallback = eligibility.for_lane(MethodLane.OBSERVATIONAL)
        if fallback is not None and fallback.state != EligibilityState.DISABLED:
            cand = _candidate(spec, fallback, primary=False)
            if not out:
                cand.warnings.append(
                    f"{primary_lane.value} requirements unmet; falling back to "
                    "covariate adjustment weakens identification"
                )
            out.append(cand)

    match = eligibility.for_lane(MethodLane.MATCHING)
    if (
        match is not None
        and match.state == EligibilityState.ENABLED
        and primary_lane == MethodLane.OBSERVATIONAL
    ):
        out.append(_candidate(spec, match, primary=False))
    return out
