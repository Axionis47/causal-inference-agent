"""Reference registration changes bookkeeping, never proposed science or authority."""

from copy import deepcopy

import pytest
from test_scientific_v8 import graph, prior, study

from backend.analysis import domain, registration
from backend.analysis.contracts import Assumption


@pytest.fixture(autouse=True)
def isolated():
    yield


def proposal():
    context, ids, target, treatments, roles = study(("t", "y"))
    value = graph(ids, target, treatments, roles, [("t", "y")], strategy="source_assignment")
    return context, prior(target, treatments, roles), value


def provenance(value, context, previous):
    return next(item for item in domain.checks(value, context, previous, scientific_version=8)
                if item.rule == "evidence_authority_and_dependencies")


def test_explicit_assumed_edge_is_registered_without_changing_the_statement_or_graph():
    context, previous, value = proposal()
    value.edges[0].basis = "assumption"
    value.edges[0].explanation = "The intervention may affect this outcome, conditional on the proposed mechanism."
    original = value.model_dump()
    assert not provenance(value, context, previous).passed
    normalized, notes = registration.normalize(value, context, previous, "causal_")
    entry = normalized.assumptions[0]
    assert entry.statement == entry.rationale == original["edges"][0]["explanation"]
    assert entry.depends_on == ["question"]
    assert entry.affects == [value.edges[0].source, value.edges[0].target]
    assert normalized.edges[0].references == ["question", "assumption:" + entry.id]
    assert normalized.edges[0].basis == "assumption" and provenance(normalized, context, previous).passed
    assert normalized.nodes == value.nodes and normalized.identifications == value.identifications
    assert value.model_dump() == original and notes
    again, more = registration.normalize(normalized, context, previous, "causal_")
    assert again == normalized and not more


def test_mixed_source_and_accepted_decision_are_separated_without_losing_citations():
    context, previous, value = proposal()
    value.edges[0].references = ["question", "decision:treatments"]
    before = domain.references(value.model_dump())
    assert not provenance(value, context, previous).passed
    normalized, notes = registration.normalize(value, context, previous, "causal_")
    assert normalized.edges[0].references == ["question"]
    assert normalized.edges[0].prerequisites == ["decision:treatments"]
    assert normalized.edges[0].basis == "supplied"
    assert domain.references(normalized.model_dump()) == before
    assert not normalized.assumptions and notes and provenance(normalized, context, previous).passed


@pytest.mark.parametrize(("basis", "refs"), [
    ("supplied", ["decision:treatments"]),
    ("supplied", ["question", "decision:unaccepted"]),
    ("supplied", ["question", "diagnostic:0"]),
    ("assumption", ["decision:treatments"]),
    ("assumption", ["question", "decision:unaccepted"]),
    ("assumption", ["question", "assumption:undeclared"]),
    ("assumption", ["invented"]),
])
def test_missing_or_mixed_authority_is_never_repaired(basis, refs):
    context, previous, value = proposal()
    value.edges[0].basis, value.edges[0].references = basis, refs
    normalized, notes = registration.normalize(value, context, previous, "causal_")
    assert normalized == value and not notes
    assert not provenance(normalized, context, previous).passed


def test_explicit_assumption_separates_accepted_dependencies_before_ledger_registration():
    context, previous, value = proposal()
    value.edges[0].basis = "assumption"
    value.edges[0].references = ["question", "decision:target", "decision:treatments"]
    original = value.model_dump()
    normalized, notes = registration.normalize(value, context, previous, "causal_")
    entry = normalized.assumptions[0]
    assert normalized.edges[0].basis == "assumption"
    assert normalized.edges[0].explanation == value.edges[0].explanation
    assert normalized.edges[0].prerequisites == ["decision:target", "decision:treatments"]
    assert normalized.edges[0].references == ["question", "assumption:" + entry.id]
    assert entry.depends_on == ["decision:target", "decision:treatments", "question"]
    assert domain.references(value.model_dump()) <= domain.references(normalized.model_dump())
    assert provenance(normalized, context, previous).passed and len(notes) == 2
    assert registration.normalize(normalized, context, previous, "causal_") == (normalized, [])
    assert value.model_dump() == original


@pytest.mark.parametrize("dependency", ["question", "diagnostic:0", "assumption:missing", "decision:unaccepted"])
def test_prerequisite_field_never_grants_authority_or_names_unaccepted_decisions(dependency):
    context, previous, value = proposal()
    value.edges[0].prerequisites = [dependency]
    assert not provenance(value, context, previous).passed


def test_existing_ledger_is_preserved_and_registration_collisions_fail_closed():
    context, previous, value = proposal()
    value.edges[0].basis = "assumption"
    normalized, _ = registration.normalize(value, context, previous, "causal_")
    conflict = normalized.assumptions[0].model_copy(update={"statement": "A different premise."})
    value.assumptions.append(conflict)
    with pytest.raises(ValueError, match="conflicts"):
        registration.normalize(value, context, previous, "causal_")
    value.assumptions = [Assumption(id="already", statement="Existing premise", rationale="Existing reason", affects=["primary"])]
    value.edges[0].references = ["question", "assumption:already"]
    normalized, notes = registration.normalize(value, context, previous, "causal_")
    assert normalized == value and not notes


def test_claim_registration_preserves_explicit_basis_and_is_scoped_to_the_task():
    context, previous, value = proposal()
    value.identifications[0].evidence.basis = "assumption"
    one, _ = registration.normalize(value, context, previous, "causal_")
    two, _ = registration.normalize(value, context, previous, "other_task_")
    assert one.assumptions[0].statement == value.identifications[0].evidence.statement
    assert one.assumptions[0].id != two.assumptions[0].id
    assert value.identifications[0].evidence.references == ["question"]


def test_registration_retains_unknown_and_withdrawn_source_failures():
    from backend.context.contracts import DatasetStatement

    context, previous, value = proposal()
    context.dataset.qualifications["withdrawn"] = DatasetStatement(
        value="Removed premise", status="withdrawn",
        source={"kind": "review", "id": "00000000-0000-0000-0000-000000000001", "author_id": "00000000-0000-0000-0000-000000000002"},
    )
    value.edges[0].basis = "assumption"
    value.edges[0].references = ["dataset.qualifications.withdrawn"]
    original = deepcopy(value)
    normalized, notes = registration.normalize(value, context, previous, "causal_")
    assert normalized == original and not notes and not provenance(normalized, context, previous).passed
