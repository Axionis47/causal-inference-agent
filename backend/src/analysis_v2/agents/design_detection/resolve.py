"""Deterministic spec refinement before eligibility rules run.

Promotions only happen when there is exactly one defensible choice; with
two or more options the field stays unresolved and the plan gate asks the
human. Every promotion is recorded so the tile can show what was decided.

This stage also folds the investigator's role table into the spec's
candidate_confounders, so the adjustment set has one source: lane eligibility
(rules) and the method plan (plan_builder) both read this list and can never
disagree about what is adjusted on.
"""
from __future__ import annotations

from src.analysis_v2.spec import (
    BANNED_ADJUSTMENT_ROLES,
    CausalSpec,
    DatasetDossier,
    ProfileSummary,
    RoleLabel,
    VariableRef,
)


def _promote_single_candidate(ref: VariableRef | None, role: str, notes: list[str]) -> None:
    if ref is None or ref.resolved or len(ref.candidates) != 1:
        return
    ref.column = ref.candidates[0]
    notes.append(f"{role}: promoted sole candidate '{ref.column}'")


def _apply_dossier_roles(
    spec: CausalSpec, dossier: DatasetDossier | None, notes: list[str]
) -> None:
    """Reconcile the investigator role table into candidate_confounders.

    Pre-treatment columns are added; columns the dossier flags with a banned
    role (post-treatment, mediator, leakage, identifier, or a structural role)
    are dropped. Treatment and outcome are never adjustment columns. After this,
    candidate_confounders is the single source both rules and plan_builder read.
    """
    if dossier is None or not dossier.roles:
        return
    protected = {spec.treatment.column, spec.outcome.column}
    banned = {r.column for r in dossier.roles if r.role in BANNED_ADJUSTMENT_ROLES}
    kept = [
        c for c in spec.candidate_confounders
        if c not in banned and c not in protected
    ]
    dropped = [c for c in spec.candidate_confounders if c not in kept]
    added: list[str] = []
    for role in dossier.roles:
        if (
            role.role == RoleLabel.PRE_TREATMENT
            and role.column not in kept
            and role.column not in protected
        ):
            kept.append(role.column)
            added.append(role.column)
    if dropped:
        notes.append(
            "adjustment set: dropped " + ", ".join(dropped) + " (investigator role)"
        )
    if added:
        notes.append(
            "adjustment set: added " + ", ".join(added)
            + " (investigator pre-treatment)"
        )
    spec.candidate_confounders = kept


def resolve_spec(
    spec: CausalSpec, profile: ProfileSummary, dossier: DatasetDossier | None = None
) -> tuple[CausalSpec, list[str]]:
    """Returns a refined copy of the spec plus the promotion notes."""
    refined = spec.model_copy(deep=True)
    notes: list[str] = []

    _promote_single_candidate(refined.outcome, "outcome", notes)
    _promote_single_candidate(refined.treatment, "treatment", notes)
    for role in ("mediator", "moderator", "instrument", "running_variable",
                 "event_column", "duration_column"):
        _promote_single_candidate(getattr(refined, role), role, notes)

    # Sole likely time column in the profile fills an unresolved time ref.
    unresolved_time = refined.time_column is None or not refined.time_column.resolved
    if unresolved_time and len(profile.likely_time_columns) == 1:
        column = profile.likely_time_columns[0]
        refined.time_column = VariableRef(
            column=column, clue=(refined.time_column.clue if refined.time_column else None)
        )
        notes.append(f"time_column: promoted sole profile candidate '{column}'")

    # Survival questions: time-to-event IS the outcome. A live intake that
    # resolved duration+event reasonably leaves outcome null; promoting it
    # here keeps the plan gate from failing a well-specified question.
    outcome_unresolved = refined.outcome is None or not refined.outcome.resolved
    if (
        outcome_unresolved
        and refined.duration_column is not None
        and refined.duration_column.resolved
        and refined.event_column is not None
        and refined.event_column.resolved
    ):
        refined.outcome = VariableRef(column=refined.duration_column.column)
        notes.append(
            f"outcome: promoted duration column '{refined.duration_column.column}' "
            "(time-to-event is the outcome of a survival question)"
        )

    # Correct the continuity flag from observed data, not the model's guess.
    if refined.treatment.resolved:
        col = profile.column(refined.treatment.column)
        if col is not None:
            is_continuous = col.semantic_type == "numeric"
            if refined.treatment_is_continuous != is_continuous:
                refined.treatment_is_continuous = is_continuous
                notes.append(
                    f"treatment_is_continuous corrected to {is_continuous} "
                    f"('{col.name}' is {col.semantic_type})"
                )

    # Fold the investigator role table in last, once treatment/outcome are
    # resolved, so the adjustment set is final before the eligibility rules run.
    _apply_dossier_roles(refined, dossier, notes)
    return refined, notes
