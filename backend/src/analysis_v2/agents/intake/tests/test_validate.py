"""Boundary validation: invented columns never reach the spec."""
from __future__ import annotations

from src.analysis_v2.agents.intake.schema import IntakeDraft
from src.analysis_v2.agents.intake.validate import to_causal_spec
from src.analysis_v2.spec import Confidence, QuestionType

LALONDE_COLUMNS = ["treat", "age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75", "re78"]


def _draft(**overrides) -> IntakeDraft:
    base = dict(
        question_type=QuestionType.BINARY_TREATMENT,
        confidence=Confidence.HIGH,
        outcome_column="re78",
        treatment_column="treat",
        reasoning_summary="Binary treatment question mapping treat to re78.",
    )
    base.update(overrides)
    return IntakeDraft(**base)


def test_a_clean_draft_passes_through_with_no_violations():
    spec, violations = to_causal_spec(_draft(), LALONDE_COLUMNS)
    assert violations == []
    assert spec.outcome.column == "re78"
    assert spec.treatment.column == "treat"
    assert spec.confidence == Confidence.HIGH


def test_an_invented_column_is_quarantined_into_a_clue_and_caps_confidence():
    spec, violations = to_causal_spec(
        _draft(outcome_column="earnings_1978"), LALONDE_COLUMNS
    )
    assert violations and "earnings_1978" in violations[0]
    assert spec.outcome.column is None
    assert "earnings_1978" in (spec.outcome.clue or "")
    assert spec.confidence == Confidence.LOW


def test_non_existent_candidates_are_dropped_but_real_ones_kept():
    spec, violations = to_causal_spec(
        _draft(
            outcome_column=None,
            outcome_candidates=["re78", "income_78", "re75"],
        ),
        LALONDE_COLUMNS,
    )
    assert spec.outcome.candidates == ["re78", "re75"]
    assert any("income_78" in v for v in violations)


def test_unresolved_outcome_caps_confidence_low_except_driver_analysis():
    spec, _ = to_causal_spec(
        _draft(outcome_column=None, outcome_clue="earnings"), LALONDE_COLUMNS
    )
    assert spec.confidence == Confidence.LOW

    driver, _ = to_causal_spec(
        _draft(
            question_type=QuestionType.DRIVER_ANALYSIS,
            outcome_column=None,
            treatment_column=None,
        ),
        LALONDE_COLUMNS,
    )
    assert driver.confidence == Confidence.HIGH


def test_derived_treatment_does_not_cap_confidence():
    spec, _ = to_causal_spec(
        _draft(
            question_type=QuestionType.DID,
            treatment_column=None,
            treatment_derived=True,
            treatment_clue="NJ stores after the wage change",
            time_column=None,
        ),
        LALONDE_COLUMNS,
    )
    assert spec.treatment.derived
    assert spec.confidence == Confidence.HIGH


def test_type_needs_surface_in_missing_info():
    rdd, _ = to_causal_spec(
        _draft(question_type=QuestionType.RDD), LALONDE_COLUMNS
    )
    assert any("running_variable" in m for m in rdd.missing_info)
    assert any("cutoff_value" in m for m in rdd.missing_info)

    iv, _ = to_causal_spec(_draft(question_type=QuestionType.IV), LALONDE_COLUMNS)
    assert any("instrument" in m for m in iv.missing_info)

    survival, _ = to_causal_spec(
        _draft(question_type=QuestionType.SURVIVAL), LALONDE_COLUMNS
    )
    assert any("duration_or_event" in m for m in survival.missing_info)
