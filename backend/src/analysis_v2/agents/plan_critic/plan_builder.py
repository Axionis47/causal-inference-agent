"""Deterministic MethodPlan construction from the selected design.

One lane, one estimator family, explicit settings. The plan critic
proposes this; S6 freezes it (possibly after user edits re-derive it).
The dossier's role table completes and vetoes the adjustment set: a
column the investigator labeled post-treatment, mediator, leakage, or
identifier never enters the covariates, and pre-treatment columns fill
in when intake named none.
"""
from __future__ import annotations

from src.analysis_v2.spec import (
    CausalSpec,
    DatasetDossier,
    DesignCandidate,
    MethodLane,
    MethodPlan,
    RoleLabel,
)

# roles that must never be adjusted on (or are already structural)
_BANNED_ROLES = {
    RoleLabel.POST_TREATMENT,
    RoleLabel.MEDIATOR,
    RoleLabel.LEAKAGE,
    RoleLabel.IDENTIFIER,
    RoleLabel.OUTCOME,
    RoleLabel.TREATMENT,
    RoleLabel.INSTRUMENT,
    RoleLabel.TIME,
    RoleLabel.GROUP,
}


def _covariates(
    spec: CausalSpec,
    exclude: list[str | None],
    dossier: DatasetDossier | None = None,
) -> list[str]:
    drop = {c for c in exclude if c}
    base = [c for c in spec.candidate_confounders if c not in drop]
    if dossier is None or not dossier.roles:
        return base
    banned = {r.column for r in dossier.roles if r.role in _BANNED_ROLES}
    out = [c for c in base if c not in banned]
    for role in dossier.roles:
        if (
            role.role == RoleLabel.PRE_TREATMENT
            and role.column not in out
            and role.column not in drop
        ):
            out.append(role.column)
    return out


def build_method_plan(
    spec: CausalSpec,
    candidate: DesignCandidate,
    dossier: DatasetDossier | None = None,
) -> MethodPlan:
    lane = candidate.lane
    outcome = spec.outcome.column or ""
    treatment = spec.treatment.column

    if lane == MethodLane.OBSERVATIONAL:
        binary = treatment is not None and not spec.treatment_is_continuous
        return MethodPlan(
            lane=lane,
            estimator="regression_adjustment",
            estimand="ate",
            outcome=outcome,
            treatment=treatment,
            covariates=_covariates(spec, [outcome, treatment], dossier),
            settings={
                "include_ipw": binary,
                "factors": spec.candidate_factors,
                "moderator": spec.moderator.column if spec.moderator else None,
            },
            rationale="covariate-adjusted regression"
            + ("; ipw cross-check on the binary treatment" if binary else ""),
        )
    if lane == MethodLane.MATCHING:
        return MethodPlan(
            lane=lane, estimator="propensity_matching", estimand="att",
            outcome=outcome, treatment=treatment,
            covariates=_covariates(spec, [outcome, treatment], dossier),
            settings={}, rationale="nearest-neighbour matching on the propensity score",
        )
    if lane == MethodLane.DID:
        return MethodPlan(
            lane=lane, estimator="difference_in_differences", estimand="att",
            outcome=outcome, treatment=treatment,
            covariates=[],
            settings={
                "time_column": spec.time_column.column if spec.time_column else None,
                "group_column": spec.group_column.column if spec.group_column else None,
                "post_column": "post",
                "needs_reshape": bool(candidate.missing_requirements),
            },
            rationale="two-by-two group x period comparison",
        )
    if lane == MethodLane.RDD:
        return MethodPlan(
            lane=lane, estimator="local_linear_rdd", estimand="late",
            outcome=outcome, treatment=treatment,
            covariates=[],
            settings={
                "running_variable": spec.running_variable.column if spec.running_variable else None,
                "cutoff": spec.cutoff_value,
                "bandwidth": None,  # auto: one sd of the running variable
            },
            rationale="local linear fit on both sides of the cutoff",
        )
    if lane == MethodLane.IV:
        return MethodPlan(
            lane=lane, estimator="two_stage_least_squares", estimand="late",
            outcome=outcome, treatment=treatment,
            covariates=_covariates(spec, [outcome, treatment], dossier),
            settings={"instrument": spec.instrument.column if spec.instrument else None},
            rationale="2sls with the supplied instrument",
        )
    if lane == MethodLane.TIME_SERIES:
        return MethodPlan(
            lane=lane, estimator="interrupted_time_series", estimand="level_shift",
            outcome=outcome, treatment=None,
            covariates=[],
            settings={
                "time_column": spec.time_column.column if spec.time_column else None,
                "intervention_date": spec.intervention_date,
            },
            rationale="segmented regression around the intervention",
        )
    if lane == MethodLane.MEDIATION:
        return MethodPlan(
            lane=lane, estimator="product_of_coefficients", estimand="indirect_effect",
            outcome=outcome, treatment=treatment,
            covariates=_covariates(spec, [outcome, treatment], dossier),
            settings={"mediator": spec.mediator.column if spec.mediator else None},
            rationale="direct/indirect decomposition with baseline adjustment",
        )
    return MethodPlan(
        lane=MethodLane.SURVIVAL, estimator="cox_proportional_hazards",
        estimand="hazard_ratio", outcome=outcome, treatment=treatment,
        covariates=_covariates(spec, [outcome, treatment], dossier),
        settings={
            "duration_column": spec.duration_column.column if spec.duration_column else None,
            "event_column": spec.event_column.column if spec.event_column else None,
        },
        rationale="cox model with kaplan-meier overview",
    )
