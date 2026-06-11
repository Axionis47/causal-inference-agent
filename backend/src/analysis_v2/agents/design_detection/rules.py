"""Deterministic lane eligibility rules.

One function per method lane; each returns a LaneEligibility with the
reason stated. A lane whose required fields are absent is DISABLED, never
silently enabled. CONDITIONAL means the structure is present and exactly
one human-confirmable item is missing (the plan gate collects it).
"""
from __future__ import annotations

import re

from src.analysis_v2.spec import (
    CausalSpec,
    EligibilityState,
    LaneEligibility,
    MethodLane,
    ProfileSummary,
    ToolEligibility,
    VariableRef,
)

_WIDE_SUFFIXES = (
    "feb", "nov", "pre", "post", "before", "after", "baseline", "followup",
    "jan", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "dec",
)


def _resolved(ref: VariableRef | None) -> bool:
    return ref is not None and ref.resolved


def detect_wide_periods(profile: ProfileSummary) -> list[str]:
    """Numeric column pairs sharing a stem with period-like suffixes
    (total_emp_feb / total_emp_nov): the wide-format pre/post signature."""
    stems: dict[str, list[str]] = {}
    for col in profile.columns:
        if col.semantic_type not in ("numeric", "ordinal"):
            continue
        m = re.match(r"^(.*?)[_\.]?(" + "|".join(_WIDE_SUFFIXES) + r"|\d{2,4})$",
                     col.name.lower())
        if m and m.group(1):
            stems.setdefault(m.group(1), []).append(col.name)
    return [name for group in stems.values() if len(group) >= 2 for name in group]


def observational(spec: CausalSpec, profile: ProfileSummary) -> LaneEligibility:
    missing = []
    if not _resolved(spec.outcome):
        missing.append("outcome")
    has_exposure = (
        _resolved(spec.treatment) or spec.treatment.derived or bool(spec.candidate_factors)
    )
    if not has_exposure:
        missing.append("treatment or candidate factors")
    if missing:
        return LaneEligibility(
            lane=MethodLane.OBSERVATIONAL, state=EligibilityState.DISABLED,
            reason=f"unresolved: {', '.join(missing)}", missing_requirements=missing,
        )
    reason = "outcome and exposure resolved"
    if not spec.candidate_confounders:
        reason += "; no confounders identified, estimates may be confounded"
    return LaneEligibility(
        lane=MethodLane.OBSERVATIONAL, state=EligibilityState.ENABLED, reason=reason
    )


def matching(spec: CausalSpec, profile: ProfileSummary) -> LaneEligibility:
    if not (_resolved(spec.outcome) and _resolved(spec.treatment)):
        return LaneEligibility(
            lane=MethodLane.MATCHING, state=EligibilityState.DISABLED,
            reason="outcome or treatment unresolved",
            missing_requirements=["resolved binary treatment", "resolved outcome"],
        )
    col = profile.column(spec.treatment.column)
    if col is None or col.semantic_type != "binary":
        kind = col.semantic_type if col else "absent"
        return LaneEligibility(
            lane=MethodLane.MATCHING, state=EligibilityState.DISABLED,
            reason=f"treatment '{spec.treatment.column}' is {kind}, not binary",
            missing_requirements=["binary treatment"],
        )
    return LaneEligibility(
        lane=MethodLane.MATCHING, state=EligibilityState.ENABLED,
        reason="binary treatment with resolved outcome",
    )


def did(spec: CausalSpec, profile: ProfileSummary) -> LaneEligibility:
    has_group = (
        _resolved(spec.group_column) or _resolved(spec.treatment) or spec.treatment.derived
    )
    if _resolved(spec.time_column) and has_group:
        return LaneEligibility(
            lane=MethodLane.DID, state=EligibilityState.ENABLED,
            reason="time and group structure present",
        )
    wide = detect_wide_periods(profile)
    if wide and has_group:
        return LaneEligibility(
            lane=MethodLane.DID, state=EligibilityState.CONDITIONAL,
            reason=f"wide pre/post columns detected ({', '.join(wide[:4])}); "
            "reshape to long format required",
            missing_requirements=["long-format time structure (reshape)"],
        )
    missing = []
    if not (_resolved(spec.time_column) or wide):
        missing.append("time structure")
    if not has_group:
        missing.append("treated/control group structure")
    return LaneEligibility(
        lane=MethodLane.DID, state=EligibilityState.DISABLED,
        reason=f"missing: {', '.join(missing)}", missing_requirements=missing,
    )


def rdd(spec: CausalSpec, profile: ProfileSummary) -> LaneEligibility:
    if not _resolved(spec.running_variable):
        return LaneEligibility(
            lane=MethodLane.RDD, state=EligibilityState.DISABLED,
            reason="no running variable detected or confirmed",
            missing_requirements=["running_variable", "cutoff_value"],
        )
    col = profile.column(spec.running_variable.column)
    if col is None or col.semantic_type not in ("numeric", "ordinal"):
        return LaneEligibility(
            lane=MethodLane.RDD, state=EligibilityState.DISABLED,
            reason=f"running variable '{spec.running_variable.column}' is not numeric",
            missing_requirements=["numeric running variable"],
        )
    if spec.cutoff_value is None:
        return LaneEligibility(
            lane=MethodLane.RDD, state=EligibilityState.CONDITIONAL,
            reason="running variable present; cutoff needs confirmation",
            missing_requirements=["cutoff_value"],
        )
    return LaneEligibility(
        lane=MethodLane.RDD, state=EligibilityState.ENABLED,
        reason=f"running variable and cutoff {spec.cutoff_value} present",
    )


def iv(spec: CausalSpec, profile: ProfileSummary) -> LaneEligibility:
    if not _resolved(spec.instrument):
        return LaneEligibility(
            lane=MethodLane.IV, state=EligibilityState.DISABLED,
            reason="no instrument explicitly supplied or strongly indicated",
            missing_requirements=["instrument"],
        )
    return LaneEligibility(
        lane=MethodLane.IV, state=EligibilityState.ENABLED,
        reason=f"instrument '{spec.instrument.column}' supplied; "
        "exclusion restriction is untestable from data alone",
    )


def time_series(spec: CausalSpec, profile: ProfileSummary) -> LaneEligibility:
    has_time = _resolved(spec.time_column) or bool(profile.likely_time_columns)
    missing = []
    if not has_time:
        missing.append("time column")
    if not _resolved(spec.outcome):
        missing.append("outcome")
    if missing:
        return LaneEligibility(
            lane=MethodLane.TIME_SERIES, state=EligibilityState.DISABLED,
            reason=f"missing: {', '.join(missing)}", missing_requirements=missing,
        )
    if spec.intervention_date is None:
        return LaneEligibility(
            lane=MethodLane.TIME_SERIES, state=EligibilityState.CONDITIONAL,
            reason="time series present; intervention date needs confirmation",
            missing_requirements=["intervention_date"],
        )
    return LaneEligibility(
        lane=MethodLane.TIME_SERIES, state=EligibilityState.ENABLED,
        reason=f"time series with intervention at {spec.intervention_date}",
    )


def mediation(spec: CausalSpec, profile: ProfileSummary) -> LaneEligibility:
    missing = []
    if not _resolved(spec.mediator):
        missing.append("mediator")
    if not (_resolved(spec.treatment) or spec.treatment.derived):
        missing.append("treatment")
    if not _resolved(spec.outcome):
        missing.append("outcome")
    if missing:
        return LaneEligibility(
            lane=MethodLane.MEDIATION, state=EligibilityState.DISABLED,
            reason=f"missing: {', '.join(missing)}", missing_requirements=missing,
        )
    return LaneEligibility(
        lane=MethodLane.MEDIATION, state=EligibilityState.ENABLED,
        reason="treatment, mediator, and outcome resolved; "
        "mediator timing is assumed, not observed",
    )


def survival(spec: CausalSpec, profile: ProfileSummary) -> LaneEligibility:
    missing = []
    if not _resolved(spec.duration_column):
        missing.append("time-to-event duration column")
    if not _resolved(spec.event_column):
        missing.append("event indicator column")
    if missing:
        return LaneEligibility(
            lane=MethodLane.SURVIVAL, state=EligibilityState.DISABLED,
            reason=f"missing: {', '.join(missing)}", missing_requirements=missing,
        )
    return LaneEligibility(
        lane=MethodLane.SURVIVAL, state=EligibilityState.ENABLED,
        reason=f"duration '{spec.duration_column.column}' and event "
        f"'{spec.event_column.column}' present",
    )


_RULES = [observational, matching, did, rdd, iv, time_series, mediation, survival]


def evaluate_all_lanes(spec: CausalSpec, profile: ProfileSummary) -> ToolEligibility:
    return ToolEligibility(lanes=[rule(spec, profile) for rule in _RULES])
