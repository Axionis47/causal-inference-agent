"""Recipe selection: which checks run for which design lane.

The base recipe always runs. The targeted recipe follows the leading
design candidate's lane; question-type extras (interaction subgroups,
factor screening) ride along when the spec carries those roles.
"""
from __future__ import annotations

from src.analysis_v2.spec import CausalSpec, MethodLane, QuestionType

from . import checks_base, checks_design, checks_groups
from .common import CheckFn

BASE: list[CheckFn] = list(checks_base.BASE_CHECKS)

TARGETED: dict[MethodLane, list[CheckFn]] = {
    MethodLane.OBSERVATIONAL: [
        checks_groups.group_outcome_comparison,
        checks_groups.covariate_balance,
        checks_groups.propensity_overlap,
        checks_groups.missingness_by_group,
    ],
    MethodLane.MATCHING: [
        checks_groups.group_outcome_comparison,
        checks_groups.covariate_balance,
        checks_groups.propensity_overlap,
        checks_groups.missingness_by_group,
    ],
    MethodLane.DID: [
        checks_design.did_trends,
        checks_design.did_pre_trends,
        checks_groups.group_outcome_comparison,
    ],
    MethodLane.RDD: [
        checks_design.rdd_around_cutoff,
        checks_design.rdd_treatment_by_side,
    ],
    MethodLane.IV: [
        checks_design.iv_first_stage,
        checks_groups.covariate_balance,
    ],
    MethodLane.TIME_SERIES: [
        checks_design.series_over_time,
    ],
    MethodLane.MEDIATION: [
        checks_design.mediation_pathways,
    ],
    MethodLane.SURVIVAL: [
        checks_design.survival_overview,
    ],
}


def _question_extras(spec: CausalSpec) -> list[CheckFn]:
    extras: list[CheckFn] = []
    if spec.question_type in (
        QuestionType.INTERACTION,
        QuestionType.HETEROGENEOUS_EFFECTS,
    ) or spec.moderator is not None:
        extras.append(checks_groups.subgroup_outcome_table)
    if spec.question_type in (
        QuestionType.MULTI_FACTOR,
        QuestionType.DRIVER_ANALYSIS,
        QuestionType.MECHANISM_SEARCH,
    ):
        extras.append(checks_groups.factor_screening)
    if spec.question_type == QuestionType.DOSE_RESPONSE or spec.treatment_is_continuous:
        extras.append(checks_design.dose_response_scatter)
    return extras


def build_recipe(spec: CausalSpec, lane: MethodLane | None) -> tuple[list[CheckFn], list[str]]:
    """(checks to run, targeted check names) for the plan artifact."""
    targeted = list(TARGETED.get(lane, [])) if lane is not None else []
    for extra in _question_extras(spec):
        if extra not in targeted:
            targeted.append(extra)
    seen: set[str] = set()
    ordered: list[CheckFn] = []
    for fn in BASE + targeted:
        if fn.__name__ not in seen:
            seen.add(fn.__name__)
            ordered.append(fn)
    return ordered, [fn.__name__ for fn in targeted]
