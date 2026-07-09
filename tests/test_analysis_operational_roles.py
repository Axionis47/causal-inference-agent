"""Operational identities and scientific roles are separate, scoped claims."""

import csv
import io

import pytest
from test_scientific_v8 import decision, failures, prior, relation, study

from backend.analysis import domain
from backend.analysis.contracts import CausalAccount, Frame


@pytest.fixture(autouse=True)
def isolated():
    yield


@pytest.mark.parametrize(("use", "role"), [
    ("unit_key", "identifier"), ("cluster_key", "identifier"),
    ("time_key", "time"), ("assignment_key", "identifier"),
    ("assignment_key", "time"), ("assignment_key", "running_variable"),
    ("assignment_key", "instrument"),
])
def test_supplied_operational_account_uses_compatible_scientific_role(use, role):
    context, _, target, treatments, roles = study()
    item = roles.assignments[0]
    item.roles = [role]
    item.operational_use = [use]
    original = roles.model_dump()
    assert failures(roles, context, prior(target, treatments, roles)) == []
    assert roles.model_dump() == original


@pytest.mark.parametrize(("use", "role"), [
    ("unit_key", "time"), ("time_key", "identifier"),
    ("cluster_key", "running_variable"), ("assignment_key", "confounder"),
    ("assignment_key", "irrelevant"),
])
def test_operational_use_does_not_grant_an_unrelated_role(use, role):
    context, _, target, treatments, roles = study()
    roles.assignments[0].roles = [role]
    roles.assignments[0].operational_use = [use]
    assert any(f"Operational use {use} needs a compatible role" in error
               for error in failures(roles, context, prior(target, treatments, roles)))


@pytest.mark.parametrize("defect", ["unknown", "assumption", "no_references", "derived_reference", "no_scope"])
def test_assignment_score_retains_evidence_and_scope_requirements(defect):
    context, _, target, treatments, roles = study()
    item = roles.assignments[0]
    item.roles, item.operational_use = ["running_variable"], ["assignment_key"]
    if defect in {"unknown", "assumption"}:
        item.account.basis = defect
    elif defect == "no_references":
        item.account.references = []
    elif defect == "derived_reference":
        item.account.references = ["decision:target"]
    else:
        item.time_scope = ""
    assert failures(roles, context, prior(target, treatments, roles))


def test_assignment_score_still_withholds_supplied_target_lineage():
    context, ids, target, treatments, roles = study()
    item = roles.assignments[0]
    item.roles, item.operational_use = ["running_variable"], ["assignment_key"]
    context.relationships["proxy"] = relation(ids["y"], ids["c"])
    frame = Frame(target=target, treatments=treatments, roles=roles,
                  causal=CausalAccount(**decision(), identification="Fixture"), graph_hash="frozen")
    projected, _ = domain.projection(frame, context, b"t,y,c\n1,9,9\n0,8,8\n", scientific_version=13)
    assert ids["c"] not in next(csv.reader(io.StringIO(projected.decode())))
