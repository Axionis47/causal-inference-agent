"""Boundary validation: the draft becomes a CausalSpec or gets quarantined.

The LLM proposes; this module disposes. Column names that do not exist in
the dataset schema are moved into clues and recorded as violations, so an
invented column can never flow downstream (BUGS_LESSONS pattern 7: pin
caller-fixed values at the handler boundary). Confidence is capped
deterministically when the mapping is shaky.
"""
from __future__ import annotations

from src.analysis_v2.spec import CausalSpec, Confidence, QuestionType, VariableRef

from .schema import IntakeDraft

# Roles a type cannot execute without; used here only to surface
# missing_info early. The authoritative eligibility rules live with
# design detection (S3).
_TYPE_NEEDS: dict[QuestionType, list[str]] = {
    QuestionType.MEDIATION: ["mediator"],
    QuestionType.IV: ["instrument"],
    QuestionType.RDD: ["running_variable", "cutoff_value"],
    QuestionType.DID: ["time_or_post_structure", "group_structure"],
    QuestionType.SURVIVAL: ["duration_or_event_column"],
    QuestionType.TIME_SERIES_INTERVENTION: ["time_column"],
    QuestionType.BEFORE_AFTER: ["time_column"],
}


def _ref(
    column: str | None,
    candidates: list[str],
    clue: str | None,
    derived: bool,
    schema_columns: set[str],
    role: str,
    violations: list[str],
) -> VariableRef:
    kept_candidates = [c for c in candidates if c in schema_columns]
    dropped = [c for c in candidates if c not in schema_columns]
    if dropped:
        violations.append(f"{role}: dropped non-existent candidates {dropped}")
    if column is not None and column not in schema_columns:
        violations.append(f"{role}: '{column}' is not a dataset column")
        clue = clue or f"model suggested '{column}', which is not a dataset column"
        column = None
    return VariableRef(column=column, candidates=kept_candidates, clue=clue, derived=derived)


def _optional_ref(
    column: str | None,
    clue: str | None,
    schema_columns: set[str],
    role: str,
    violations: list[str],
) -> VariableRef | None:
    if column is None and not clue:
        return None
    return _ref(column, [], clue, False, schema_columns, role, violations)


def to_causal_spec(
    draft: IntakeDraft, schema_columns: list[str]
) -> tuple[CausalSpec, list[str]]:
    """Validate a draft against the real schema. Returns (spec, violations)."""
    cols = set(schema_columns)
    violations: list[str] = []

    outcome = _ref(
        draft.outcome_column,
        draft.outcome_candidates,
        draft.outcome_clue,
        False,
        cols,
        "outcome",
        violations,
    )
    treatment = _ref(
        draft.treatment_column,
        draft.treatment_candidates,
        draft.treatment_clue,
        draft.treatment_derived,
        cols,
        "treatment",
        violations,
    )

    missing_info = list(draft.missing_info)
    spec = CausalSpec(
        question_type=draft.question_type,
        type_candidates=[t for t in draft.type_candidates if t != draft.question_type],
        confidence=draft.confidence,
        outcome=outcome,
        treatment=treatment,
        mediator=_optional_ref(draft.mediator_column, draft.mediator_clue, cols, "mediator", violations),
        moderator=_optional_ref(draft.moderator_column, draft.moderator_clue, cols, "moderator", violations),
        instrument=_optional_ref(draft.instrument_column, draft.instrument_clue, cols, "instrument", violations),
        running_variable=_optional_ref(
            draft.running_variable_column, draft.running_variable_clue, cols, "running_variable", violations
        ),
        cutoff_value=draft.cutoff_value,
        time_column=_optional_ref(draft.time_column, None, cols, "time_column", violations),
        group_column=_optional_ref(draft.group_column, None, cols, "group_column", violations),
        event_column=_optional_ref(draft.event_column, None, cols, "event_column", violations),
        duration_column=_optional_ref(draft.duration_column, None, cols, "duration_column", violations),
        intervention_date=draft.intervention_date,
        candidate_factors=[c for c in draft.candidate_factors if c in cols],
        candidate_confounders=[c for c in draft.candidate_confounders if c in cols],
        treatment_is_continuous=draft.treatment_is_continuous,
        missing_info=missing_info,
        assumptions=list(draft.assumptions),
        notes=draft.notes,
    )

    _surface_type_needs(spec, missing_info)
    spec.missing_info = missing_info
    spec.confidence = _capped_confidence(spec, violations)
    return spec, violations


def _surface_type_needs(spec: CausalSpec, missing_info: list[str]) -> None:
    needs = _TYPE_NEEDS.get(spec.question_type, [])
    checks = {
        "mediator": spec.mediator is not None and spec.mediator.resolved,
        "instrument": spec.instrument is not None and spec.instrument.resolved,
        "running_variable": spec.running_variable is not None and spec.running_variable.resolved,
        "cutoff_value": spec.cutoff_value is not None,
        "time_or_post_structure": (spec.time_column is not None and spec.time_column.resolved)
        or spec.treatment.derived,
        "group_structure": (spec.group_column is not None and spec.group_column.resolved)
        or spec.treatment.resolved
        or spec.treatment.derived,
        "duration_or_event_column": (
            spec.duration_column is not None and spec.duration_column.resolved
        )
        or (spec.event_column is not None and spec.event_column.resolved),
        "time_column": spec.time_column is not None and spec.time_column.resolved,
    }
    for need in needs:
        if not checks.get(need, True):
            note = f"{need} is required for a {spec.question_type.value} question but unresolved"
            if note not in missing_info:
                missing_info.append(note)


def _capped_confidence(spec: CausalSpec, violations: list[str]) -> Confidence:
    if violations:
        return Confidence.LOW
    if not spec.outcome.resolved and spec.question_type != QuestionType.DRIVER_ANALYSIS:
        return Confidence.LOW
    needs_treatment = spec.question_type not in (
        QuestionType.DRIVER_ANALYSIS,
        QuestionType.MECHANISM_SEARCH,
        QuestionType.MULTI_FACTOR,
        QuestionType.TIME_SERIES_INTERVENTION,
        QuestionType.BEFORE_AFTER,
    )
    if needs_treatment and not (spec.treatment.resolved or spec.treatment.derived):
        return min(spec.confidence, Confidence.MEDIUM, key=_confidence_rank)
    return spec.confidence


def _confidence_rank(value: Confidence) -> int:
    return {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}[value]
