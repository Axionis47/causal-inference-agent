"""dag_from_dossier projects the role table into a CausalDAG whose backdoor
adjustment set matches the reconciled confounder list (the fold S3 performs)."""
from __future__ import annotations

from src.analysis_v2.agents.investigator.dag import dag_from_dossier
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
        _dossier([
            ("treat", RoleLabel.TREATMENT), ("re78", RoleLabel.OUTCOME),
            ("age", RoleLabel.PRE_TREATMENT), ("educ", RoleLabel.PRE_TREATMENT),
        ]),
        _spec(),
    )
    z = dag.adjustment_set()
    assert z == {"age", "educ"}
    assert dag.treatment == "treat" and dag.outcome == "re78"


def test_intake_confounders_are_included():
    dag = dag_from_dossier(
        _dossier([("treat", RoleLabel.TREATMENT), ("re78", RoleLabel.OUTCOME)]),
        _spec(confounders=("age",)),
    )
    z = dag.adjustment_set()
    assert z == {"age"}


def test_a_mediator_is_in_the_dag_but_not_adjusted_for():
    dag = dag_from_dossier(
        _dossier([
            ("treat", RoleLabel.TREATMENT), ("re78", RoleLabel.OUTCOME),
            ("m", RoleLabel.MEDIATOR),
        ]),
        _spec(),
    )
    z = dag.adjustment_set()
    assert z == set()  # the mediator is a descendant of the treatment
    assert any(n.name == "m" for n in dag.nodes)


def test_a_banned_role_is_excluded_from_the_backdoor_set():
    # intake guessed 'leaky' a confounder; the dossier flags it post-treatment.
    dag = dag_from_dossier(
        _dossier([
            ("treat", RoleLabel.TREATMENT), ("re78", RoleLabel.OUTCOME),
            ("leaky", RoleLabel.POST_TREATMENT),
        ]),
        _spec(confounders=("leaky", "age")),
    )
    z = dag.adjustment_set()
    assert z == {"age"}
