"""Deterministic spec refinement before eligibility rules run.

Promotions only happen when there is exactly one defensible choice; with
two or more options the field stays unresolved and the plan gate asks the
human. Every promotion is recorded so the tile can show what was decided.

The adjustment set is no longer reconciled here: design_detection builds the
causal DAG from the refined spec and dossier (dag.py) and reads the backdoor
adjustment set off it, so identification has a single source.
"""
from __future__ import annotations

from src.analysis_v2.spec import CausalSpec, ProfileSummary, VariableRef


def _promote_single_candidate(ref: VariableRef | None, role: str, notes: list[str]) -> None:
    if ref is None or ref.resolved or len(ref.candidates) != 1:
        return
    ref.column = ref.candidates[0]
    notes.append(f"{role}: promoted sole candidate '{ref.column}'")


def resolve_spec(
    spec: CausalSpec, profile: ProfileSummary
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

    return refined, notes
