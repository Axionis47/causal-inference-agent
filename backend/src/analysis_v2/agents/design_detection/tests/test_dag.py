"""dag_from_dossier / apply_dag_adjustment at S3: the DAG is built from the
refined spec and dossier, and its backdoor adjustment set becomes the spec's
candidate_confounders, matching the fold design_detection used to do inline."""
from __future__ import annotations

from src.analysis_v2.agents.design_detection.dag import (
    apply_dag_adjustment,
    dag_from_dossier,
)
from src.analysis_v2.spec import (
    CausalSpec,
    ColumnRole,
    DatasetDossier,
    QuestionType,
    RoleLabel,
    VariableRef,
)


def _dossier(roles: list[tuple[str, RoleLabel]]) -> DatasetDossier:
    return DatasetDossier(
        provenance="t",
        roles=[ColumnRole(column=c, role=r, reason="x") for c, r in roles],
        summary="s",
    )


def _spec(confounders: tuple[str, ...] = ()) -> CausalSpec:
    return CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        treatment=VariableRef(column="treat"),
        outcome=VariableRef(column="re78"),
        candidate_confounders=list(confounders),
    )


def test_pre_treatment_roles_become_the_backdoor_set():
    dag = dag_from_dossier(
        _spec(),
        _dossier([
            ("treat", RoleLabel.TREATMENT), ("re78", RoleLabel.OUTCOME),
            ("age", RoleLabel.PRE_TREATMENT), ("educ", RoleLabel.PRE_TREATMENT),
        ]),
    )
    assert dag.adjustment_set() == {"age", "educ"}
    assert dag.treatment == "treat" and dag.outcome == "re78"


def test_intake_confounders_are_included():
    dag = dag_from_dossier(
        _spec(confounders=("age",)),
        _dossier([("treat", RoleLabel.TREATMENT), ("re78", RoleLabel.OUTCOME)]),
    )
    assert dag.adjustment_set() == {"age"}


def test_a_mediator_is_in_the_dag_but_not_adjusted_for():
    dag = dag_from_dossier(
        _spec(),
        _dossier([
            ("treat", RoleLabel.TREATMENT), ("re78", RoleLabel.OUTCOME),
            ("m", RoleLabel.MEDIATOR),
        ]),
    )
    assert dag.adjustment_set() == set()  # the mediator is a descendant of treatment
    assert any(n.name == "m" for n in dag.nodes)


def test_a_banned_role_is_excluded_from_the_backdoor_set():
    # intake guessed 'leaky' a confounder; the dossier flags it post-treatment.
    dag = dag_from_dossier(
        _spec(confounders=("leaky", "age")),
        _dossier([
            ("treat", RoleLabel.TREATMENT), ("re78", RoleLabel.OUTCOME),
            ("leaky", RoleLabel.POST_TREATMENT),
        ]),
    )
    assert dag.adjustment_set() == {"age"}


def test_apply_dag_adjustment_sets_candidate_confounders_from_the_backdoor_set():
    spec = _spec(confounders=("age", "educ"))
    dag = apply_dag_adjustment(
        spec,
        _dossier([("treat", RoleLabel.TREATMENT), ("re78", RoleLabel.OUTCOME)]),
    )
    assert spec.candidate_confounders == ["age", "educ"]  # sorted backdoor set
    assert dag.adjustment_set() == {"age", "educ"}
