"""Plan gate rules: auto-approve, ask the user, or fail with the missing list.

Hard-missing required info fails the gate. Confirmation is required for
every uncertainty a human can actually resolve: low/medium confidence,
derived treatment, unconfirmed cutoff / intervention date / reshape,
competing designs, or candidate-only role mappings. Auto-approval is the
exception, earned only by a fully-resolved high-confidence plan.
"""
from __future__ import annotations

from src.analysis_v2.spec import (
    CausalSpec,
    Confidence,
    ConfirmationCard,
    ConfirmationItem,
    DesignCandidate,
    EligibilityState,
    QuestionType,
    ToolEligibility,
)


def hard_missing(spec: CausalSpec, candidates: list[DesignCandidate]) -> list[str]:
    missing: list[str] = []
    if not spec.outcome.resolved and not spec.outcome.candidates:
        if spec.question_type != QuestionType.DRIVER_ANALYSIS:
            missing.append("outcome: no column and no candidates")
    if not candidates:
        missing.append("no executable design candidate for this question")
    return missing


def confirmation_items(
    spec: CausalSpec,
    candidates: list[DesignCandidate],
    eligibility: ToolEligibility | None,
) -> list[ConfirmationItem]:
    items: list[ConfirmationItem] = []
    primary = candidates[0] if candidates else None

    def _role_item(ref, field: str, label: str, why: str) -> None:
        if ref is not None and not ref.resolved and ref.candidates:
            items.append(
                ConfirmationItem(
                    field=field, label=label,
                    current_value=ref.candidates[0],
                    options=ref.candidates, required=True, why=why,
                )
            )

    _role_item(
        spec.outcome, "outcome", "Outcome column",
        "Several columns could be the outcome; pick the one your question asks about.",
    )
    _role_item(
        spec.treatment, "treatment", "Treatment column",
        "Several columns could be the treatment or exposure.",
    )

    if primary is not None and primary.lane.value == "rdd" and spec.cutoff_value is None:
        items.append(
            ConfirmationItem(
                field="cutoff_value", label="Assignment cutoff",
                current_value=None, options=[], required=True,
                why="RDD needs the exact threshold that assigns treatment; "
                "it was not stated and must not be guessed.",
            )
        )
    if (
        primary is not None
        and primary.lane.value == "time_series"
        and spec.intervention_date is None
    ):
        items.append(
            ConfirmationItem(
                field="intervention_date", label="Intervention date (ISO)",
                current_value=None, options=[], required=True,
                why="The interrupted-time-series split needs the date the change happened.",
            )
        )
    if primary is not None and primary.missing_requirements:
        for req in primary.missing_requirements:
            if "reshape" in req:
                items.append(
                    ConfirmationItem(
                        field="confirm_reshape", label="Confirm wide-to-long reshape",
                        current_value="yes", options=["yes", "no"], required=True,
                        why="Pre/post outcome columns were detected in wide format; "
                        "the lane reshapes them into period rows.",
                    )
                )
    if spec.treatment.derived:
        items.append(
            ConfirmationItem(
                field="confirm_derived_treatment", label="Confirm derived treatment",
                current_value="yes", options=["yes", "no"], required=True,
                why=f"The treatment is constructed, not a column: "
                f"{spec.treatment.clue or 'derived indicator'}.",
            )
        )
    if len(candidates) > 1 and candidates[1].confidence == candidates[0].confidence:
        items.append(
            ConfirmationItem(
                field="lane", label="Design lane",
                current_value=candidates[0].lane.value,
                options=[c.lane.value for c in candidates], required=True,
                why="More than one design fits about equally well.",
            )
        )
    return items


def needs_confirmation_reasons(
    spec: CausalSpec, candidates: list[DesignCandidate]
) -> list[str]:
    reasons: list[str] = []
    if spec.confidence != Confidence.HIGH:
        reasons.append(f"question mapping confidence is {spec.confidence.value}")
    primary = candidates[0] if candidates else None
    if primary is not None and primary.confidence != Confidence.HIGH:
        reasons.append(f"design confidence is {primary.confidence.value}")
    if primary is not None and primary.missing_requirements:
        reasons.append(
            "design requirements need confirmation: "
            + ", ".join(primary.missing_requirements)
        )
    if spec.treatment.derived:
        reasons.append("treatment is derived, not an existing column")
    if spec.missing_info:
        reasons.append("intake flagged missing information: " + "; ".join(spec.missing_info[:3]))
    return reasons


def build_card(
    spec: CausalSpec,
    candidates: list[DesignCandidate],
    items: list[ConfirmationItem],
    reasons: list[str],
) -> ConfirmationCard:
    primary = candidates[0]
    summary = (
        f"Question type: {spec.question_type.value}. Design: {primary.design_label}. "
        f"Outcome: {spec.outcome.column or spec.outcome.candidates or 'unresolved'}. "
        f"Treatment: {spec.treatment.column or ('derived: ' + (spec.treatment.clue or ''))}. "
        f"Why confirmation is needed: {'; '.join(reasons) or 'final check before execution'}."
    )
    return ConfirmationCard(
        headline=f"Confirm the {primary.lane.value} plan before it runs",
        plan_summary=summary[:2000],
        items=items,
    )
