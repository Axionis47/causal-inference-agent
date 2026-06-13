"""resolve_spec folds the investigator role table into the adjustment set.

After S3, candidate_confounders is the single source both lane eligibility
(rules) and the method plan (plan_builder) read, so the design narrative and
the executed covariates can never contradict each other.
"""
from __future__ import annotations

from src.analysis_v2.agents.design_detection.resolve import resolve_spec
from src.analysis_v2.agents.design_detection.rules import observational
from src.analysis_v2.agents.plan_critic.plan_builder import build_method_plan
from src.analysis_v2.spec import (
    CausalSpec,
    ColumnProfile,
    ColumnRole,
    Confidence,
    DatasetDossier,
    DesignCandidate,
    MethodLane,
    ProfileSummary,
    QuestionType,
    RoleLabel,
    VariableRef,
)


def _profile() -> ProfileSummary:
    cols = [
        ColumnProfile(
            name=n, dtype="float64", semantic_type=t,
            missing_count=0, missing_fraction=0.0, n_unique=(2 if t == "binary" else 50),
        )
        for n, t in [
            ("treat", "binary"), ("re78", "numeric"), ("age", "numeric"),
            ("re74", "numeric"), ("re75", "numeric"), ("total ads", "numeric"),
            ("row_id", "numeric"),
        ]
    ]
    return ProfileSummary(n_rows=100, n_columns=len(cols), columns=cols)


PROFILE = _profile()


def _spec(confounders: list[str]) -> CausalSpec:
    return CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="re78"),
        treatment=VariableRef(column="treat"),
        candidate_confounders=confounders,
    )


def _dossier(roles: list[tuple[str, RoleLabel]]) -> DatasetDossier:
    return DatasetDossier(
        provenance="test",
        roles=[ColumnRole(column=c, role=r, reason="investigated") for c, r in roles],
        summary="test dossier",
    )


def _candidate() -> DesignCandidate:
    return DesignCandidate(
        lane=MethodLane.OBSERVATIONAL, design_label="observational",
        confidence=Confidence.HIGH, rationale="resolved",
    )


def test_pre_treatment_roles_fold_into_an_empty_adjustment_set():
    """The live-lalonde case: intake named no confounders; the dossier knew
    all the pre-treatment roles. The fold now happens at S3, not S5."""
    refined, notes = resolve_spec(
        _spec([]),
        PROFILE,
        _dossier([
            ("treat", RoleLabel.TREATMENT), ("re78", RoleLabel.OUTCOME),
            ("age", RoleLabel.PRE_TREATMENT), ("re74", RoleLabel.PRE_TREATMENT),
            ("re75", RoleLabel.PRE_TREATMENT),
        ]),
    )
    assert refined.candidate_confounders == ["age", "re74", "re75"]
    assert any("added" in n for n in notes)


def test_banned_roles_drop_from_candidate_confounders():
    refined, notes = resolve_spec(
        _spec(["total ads", "row_id", "age"]),
        PROFILE,
        _dossier([
            ("total ads", RoleLabel.POST_TREATMENT),
            ("row_id", RoleLabel.IDENTIFIER),
            ("age", RoleLabel.PRE_TREATMENT),
        ]),
    )
    assert refined.candidate_confounders == ["age"]
    assert any("dropped" in n for n in notes)


def test_treatment_and_outcome_never_become_confounders():
    """A dossier that mislabels the treatment pre_treatment must not get it
    adjusted on; treatment/outcome are protected regardless of role."""
    refined, _ = resolve_spec(
        _spec([]),
        PROFILE,
        _dossier([
            ("treat", RoleLabel.PRE_TREATMENT), ("age", RoleLabel.PRE_TREATMENT),
        ]),
    )
    assert "treat" not in refined.candidate_confounders
    assert refined.candidate_confounders == ["age"]


def test_no_dossier_leaves_candidate_confounders_unchanged():
    refined, _ = resolve_spec(_spec(["age", "re74"]), PROFILE, None)
    assert refined.candidate_confounders == ["age", "re74"]


def test_eligibility_message_and_method_plan_agree_when_dossier_supplies_confounders():
    """Refactor A regression: S3's 'no confounders' verdict and the S5
    covariate list read one source and can no longer contradict."""
    refined, _ = resolve_spec(
        _spec([]),
        PROFILE,
        _dossier([
            ("treat", RoleLabel.TREATMENT), ("re78", RoleLabel.OUTCOME),
            ("age", RoleLabel.PRE_TREATMENT), ("re74", RoleLabel.PRE_TREATMENT),
        ]),
    )
    elig = observational(refined, PROFILE)
    assert "no confounders" not in elig.reason

    plan = build_method_plan(refined, _candidate(), _dossier([
        ("treat", RoleLabel.TREATMENT), ("re78", RoleLabel.OUTCOME),
        ("age", RoleLabel.PRE_TREATMENT), ("re74", RoleLabel.PRE_TREATMENT),
    ]))
    assert plan.covariates == ["age", "re74"]


def test_truly_no_confounders_reports_none_and_adjusts_for_none():
    refined, _ = resolve_spec(
        _spec([]),
        PROFILE,
        _dossier([("treat", RoleLabel.TREATMENT), ("re78", RoleLabel.OUTCOME)]),
    )
    elig = observational(refined, PROFILE)
    assert "no confounders" in elig.reason
    plan = build_method_plan(refined, _candidate(), None)
    assert plan.covariates == []
