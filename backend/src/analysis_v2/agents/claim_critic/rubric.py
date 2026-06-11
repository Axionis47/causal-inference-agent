"""The claim-strength rubric: design base, robustness adjustment, caps.

Deterministic. Exploratory question types can never rise above
exploratory; a failed assumption check pins not_supported; a confidence
interval that spans zero converts the claim into a no-clear-effect
statement rather than a weaker yes.
"""
from __future__ import annotations

from src.analysis_v2.spec import (
    ClaimStrength,
    EstimateResult,
    MethodLane,
    QuestionType,
    RobustnessStatus,
    SensitivityResult,
)

_BASE: dict[MethodLane, ClaimStrength] = {
    MethodLane.RDD: ClaimStrength.MODERATE,
    MethodLane.DID: ClaimStrength.MODERATE,
    MethodLane.IV: ClaimStrength.MODERATE,
    MethodLane.OBSERVATIONAL: ClaimStrength.WEAK,
    MethodLane.MATCHING: ClaimStrength.WEAK,
    MethodLane.SURVIVAL: ClaimStrength.WEAK,
    MethodLane.TIME_SERIES: ClaimStrength.WEAK,
    MethodLane.MEDIATION: ClaimStrength.WEAK,
}

_ORDER = [
    ClaimStrength.NOT_SUPPORTED,
    ClaimStrength.EXPLORATORY,
    ClaimStrength.WEAK,
    ClaimStrength.MODERATE,
    ClaimStrength.STRONG,
]

_EXPLORATORY_TYPES = {
    QuestionType.DRIVER_ANALYSIS,
    QuestionType.MECHANISM_SEARCH,
    QuestionType.MULTI_FACTOR,
}

ALLOWED_LANGUAGE = [
    "consistent with",
    "suggests",
    "estimated effect",
    "under these assumptions",
    "associated with",
]
FORBIDDEN_LANGUAGE = [
    "proves",
    "guarantees",
    "definitely causes",
    "causal truth",
    "demonstrates conclusively",
]


def _step(strength: ClaimStrength, delta: int) -> ClaimStrength:
    index = _ORDER.index(strength) + delta
    index = max(1, min(index, len(_ORDER) - 1))  # never step into not_supported
    return _ORDER[index]


def claim_strength(
    question_type: QuestionType,
    result: EstimateResult,
    sensitivity: SensitivityResult | None,
) -> tuple[ClaimStrength, list[str]]:
    """Returns (strength, rationale notes)."""
    notes: list[str] = []
    if sensitivity is not None and sensitivity.robustness == RobustnessStatus.NOT_SUPPORTED:
        return (
            ClaimStrength.NOT_SUPPORTED,
            ["a design assumption failed its diagnostic; the estimate cannot "
             "support a causal claim"],
        )

    strength = _BASE[result.lane]
    notes.append(f"base strength for the {result.lane.value} design: {strength.value}")

    if sensitivity is not None:
        if sensitivity.robustness == RobustnessStatus.ROBUST:
            strength = _step(strength, +1)
            notes.append("upgraded one step: stable across the sensitivity checks")
        elif sensitivity.robustness == RobustnessStatus.FRAGILE:
            strength = _step(strength, -1)
            notes.append("downgraded one step: fragile under sensitivity checks")

    if question_type in _EXPLORATORY_TYPES:
        strength = ClaimStrength.EXPLORATORY
        notes.append(
            f"{question_type.value} questions are exploratory by nature; "
            "no per-factor causal claims"
        )
    if question_type == QuestionType.BEFORE_AFTER:
        strength = min(strength, ClaimStrength.WEAK, key=_ORDER.index)
        notes.append("before/after comparison without a control group caps at weak")

    primary = result.primary
    if (
        primary.ci_lower is not None
        and primary.ci_upper is not None
        and primary.ci_lower <= 0 <= primary.ci_upper
    ):
        notes.append(
            "the confidence interval spans zero: the data do not show a clear "
            "effect; phrase findings as such"
        )
    return strength, notes


def limitations(
    question_type: QuestionType,
    result: EstimateResult,
    sensitivity: SensitivityResult | None,
) -> list[str]:
    out: list[str] = []
    lane = result.lane
    if lane in (MethodLane.OBSERVATIONAL, MethodLane.MATCHING, MethodLane.SURVIVAL):
        out.append(
            "unmeasured confounding can bias the estimate; only measured "
            "covariates were adjusted"
        )
    if lane == MethodLane.IV:
        out.append(
            "the exclusion restriction cannot be tested from data alone; the "
            "estimate applies to instrument compliers"
        )
    if lane == MethodLane.MEDIATION:
        out.append("mediator timing is assumed, not observed")
    if lane == MethodLane.TIME_SERIES or question_type == QuestionType.BEFORE_AFTER:
        out.append(
            "no control group: anything else that changed around the "
            "intervention is an alternative explanation"
        )
    if lane == MethodLane.RDD:
        out.append("the estimate is local to units near the cutoff")
    if lane == MethodLane.DID:
        out.append("parallel pre-trends are assumed; with two periods they are untestable")
    if question_type in _EXPLORATORY_TYPES:
        out.append(
            "associations from a screening model rank candidates; they are "
            "not causal effect estimates"
        )
    if sensitivity is not None:
        out.extend(
            f"sensitivity: {c.detail}"
            for c in sensitivity.checks
            if c.status.value in ("warning", "fail")
        )
    out.extend(f"estimation: {w}" for w in result.warnings[:3])
    return out
