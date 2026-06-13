"""Deterministic MethodPlan construction from the selected design.

One lane, one estimator family, explicit settings. The plan critic
proposes this; S6 freezes it (possibly after user edits re-derive it).
The adjustment set is the candidate_confounders list design_detection (S3)
already reconciled from the investigator's role table; _covariates only
re-asserts the banned-role veto, so a structural or post-treatment column
can never slip in even if an upstream change regresses.
"""
from __future__ import annotations

import pandas as pd

from src.analysis_v2.spec import (
    BANNED_ADJUSTMENT_ROLES,
    CausalSpec,
    DatasetDossier,
    DesignCandidate,
    MethodLane,
    MethodPlan,
)


def _covariates(
    spec: CausalSpec,
    exclude: list[str | None],
    dossier: DatasetDossier | None = None,
) -> list[str]:
    """Read the adjustment set S3 folded into candidate_confounders. The
    banned-role filter is a defensive assertion: with one source it is a no-op,
    but it guarantees a banned column can never reach the covariates."""
    drop = {c for c in exclude if c}
    if dossier is not None and dossier.roles:
        drop |= {r.column for r in dossier.roles if r.role in BANNED_ADJUSTMENT_ROLES}
    return [c for c in spec.candidate_confounders if c not in drop]


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


def select_runnable(
    spec: CausalSpec,
    candidates: list[DesignCandidate],
    dossier: DatasetDossier | None,
    frame: pd.DataFrame,
) -> tuple[DesignCandidate, MethodPlan, list[str]]:
    """Pick the highest-ranked candidate whose plan runs cleanly on the data.

    The "fix things" step: when the primary design is too thin to estimate
    (a check_ready failure on the real frame), fall down the ranking to the
    first design that does run, and note the switch. When nothing runs cleanly,
    the top candidate's plan is returned with a note so the readiness gate (S6b)
    reports the blocking reason rather than the design being dropped silently.
    Only the auto-approve path calls this; designs still awaiting user
    confirmation are left untouched until their open items are resolved.
    """
    from src.analysis_v2.agents.method_lane.lanes import LANE_CHECKS

    top = candidates[0]
    top_plan = build_method_plan(spec, top, dossier)
    for candidate in candidates:
        plan = top_plan if candidate is top else build_method_plan(spec, candidate, dossier)
        check = LANE_CHECKS.get(plan.lane)
        try:
            blocking = list(check(frame, plan, spec)) if check is not None else []
        except Exception as exc:  # a check that raises means this lane cannot run
            blocking = [f"{plan.lane.value} readiness check raised: {exc}"]
        if not blocking:
            if candidate is top:
                return candidate, plan, []
            return candidate, plan, [
                f"primary design '{top.lane.value}' is not runnable on this data; "
                f"using '{candidate.lane.value}' instead"
            ]
    return top, top_plan, [
        f"no design candidate runs cleanly on this data; keeping "
        f"'{top.lane.value}' (the readiness check explains why)"
    ]
