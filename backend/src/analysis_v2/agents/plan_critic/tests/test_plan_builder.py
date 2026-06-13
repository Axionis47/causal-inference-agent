"""plan_builder reads the adjustment set design_detection (S3) folded into
candidate_confounders, and re-asserts the banned-role veto defensively."""
from __future__ import annotations

from src.analysis_v2.agents.plan_critic.plan_builder import build_method_plan
from src.analysis_v2.spec import (
    CausalSpec,
    ColumnRole,
    Confidence,
    DatasetDossier,
    DesignCandidate,
    MethodLane,
    QuestionType,
    RoleLabel,
    VariableRef,
)


def _candidate(lane: MethodLane = MethodLane.OBSERVATIONAL) -> DesignCandidate:
    return DesignCandidate(
        lane=lane, design_label="covariate-adjusted observational comparison",
        confidence=Confidence.HIGH, rationale="resolved",
    )


def _dossier(roles: list[tuple[str, RoleLabel]]) -> DatasetDossier:
    return DatasetDossier(
        provenance="test",
        roles=[
            ColumnRole(column=c, role=r, reason="investigated") for c, r in roles
        ],
        summary="test dossier",
    )


def test_plan_reads_the_adjustment_set_design_detection_folded():
    """S3 reconciles the dossier into candidate_confounders; plan_builder reads
    that single source rather than re-deriving from the role table (the fold
    itself is covered in design_detection/tests/test_resolve.py)."""
    spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(column="treat"),
        candidate_confounders=["age", "re74", "re75"],  # already folded by S3
    )
    dossier = _dossier(
        [
            ("treat", RoleLabel.TREATMENT),
            ("re78", RoleLabel.OUTCOME),
            ("age", RoleLabel.PRE_TREATMENT),
            ("re74", RoleLabel.PRE_TREATMENT),
            ("re75", RoleLabel.PRE_TREATMENT),
        ]
    )
    plan = build_method_plan(spec, _candidate(), dossier)
    assert plan.covariates == ["age", "re74", "re75"]


def test_dossier_vetoes_a_post_treatment_column_intake_listed():
    """The marketing-ab trap: 'total ads' is post-treatment exposure; it
    must leave the adjustment set even when intake offered it."""
    spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="converted"),
        treatment=VariableRef(column="test group"),
        candidate_confounders=["total ads", "most ads day"],
    )
    dossier = _dossier(
        [
            ("total ads", RoleLabel.POST_TREATMENT),
            ("most ads day", RoleLabel.POST_TREATMENT),
        ]
    )
    plan = build_method_plan(spec, _candidate(), dossier)
    assert plan.covariates == []


def test_without_a_dossier_the_spec_confounders_pass_through_unchanged():
    spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="y"),
        treatment=VariableRef(column="t"),
        candidate_confounders=["a", "b"],
    )
    plan = build_method_plan(spec, _candidate(), None)
    assert plan.covariates == ["a", "b"]


def test_identifier_and_leakage_roles_never_enter_the_covariates():
    spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="y"),
        treatment=VariableRef(column="t"),
        candidate_confounders=["row_id", "y_copy", "a"],
    )
    dossier = _dossier(
        [
            ("row_id", RoleLabel.IDENTIFIER),
            ("y_copy", RoleLabel.LEAKAGE),
            ("a", RoleLabel.PRE_TREATMENT),
        ]
    )
    plan = build_method_plan(spec, _candidate(), dossier)
    assert plan.covariates == ["a"]
