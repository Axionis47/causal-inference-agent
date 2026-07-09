"""Local Wald identification needs its explicit conditional scientific obligations."""

import pytest
from test_analysis_contrasts import accepted, discontinuity
from test_scientific_v8 import claim, graph, prior

from backend.analysis import domain
from backend.analysis.contracts import Assumption, IdentificationObligation


@pytest.fixture(autouse=True)
def isolated():
    yield


def account():
    context, ids, target, treatments, roles = discontinuity(True)
    treatments = accepted(treatments, context, target)
    causal = graph(ids, target, treatments, roles, [("score", "t"), ("t", "y")], strategy="discontinuity")
    causal.identifications[0].obligations = [IdentificationObligation(name=name, claim=claim(statement)) for name, statement in (
        ("cutoff_rule", "The recorded eligibility rule changes at the fixed cutoff."),
        ("continuity", "Potential outcomes and compliance-type distributions are locally continuous at the cutoff."),
        ("exclusion", "Eligibility has no discontinuous outcome effect at the cutoff except through receipt; smooth score effects remain allowed."),
        ("monotonicity", "Eligibility does not reduce receipt for units in the local target."),
        ("relevance", "The local receipt probability has a nonzero discontinuity at the cutoff."),
    )]
    return context, target, treatments, roles, causal


def checks(fixture, version=15):
    context, target, treatments, roles, causal = fixture
    return domain.checks(causal, context, prior(target, treatments, roles), scientific_version=version)


def test_fuzzy_conditional_obligations_pass_without_promoting_assumptions_to_supplied_facts():
    fixture = account()
    causal = fixture[-1]
    obligation = next(item for item in causal.identifications[0].obligations if item.name == "exclusion")
    causal.assumptions = [Assumption(id="local_exclusion", statement=obligation.claim.statement,
                                    rationale="Required local receipt-effect premise, not established by observed jump values.", affects=["primary"])]
    obligation.claim.basis = "assumption"
    obligation.claim.references = ["assumption:local_exclusion"]
    before = causal.model_dump()
    result = checks(fixture)
    assert all(item.passed for item in result), [(item.rule, item.detail) for item in result if not item.passed]
    assert causal.model_dump() == before


@pytest.mark.parametrize("name", ["exclusion", "monotonicity", "relevance"])
@pytest.mark.parametrize("defect", ["missing", "unknown"])
def test_missing_or_unknown_fuzzy_obligations_cannot_certify_local_wald(name, defect):
    fixture = account()
    obligations = fixture[-1].identifications[0].obligations
    if defect == "missing":
        obligations[:] = [item for item in obligations if item.name != name]
    else:
        obligation = next(item for item in obligations if item.name == name)
        obligation.claim.basis = "unknown"
        obligation.claim.references = []
    assert any(not item.passed and item.rule == "causal_graph_consistency" for item in checks(fixture))


def test_sharp_outcome_jump_does_not_acquire_receipt_exclusion_or_monotonicity_requirements():
    fixture = account()
    fixture[2].effects[0].comparison.discontinuity.estimand = "outcome_jump"
    fixture[-1].identifications[0].obligations = fixture[-1].identifications[0].obligations[:2]
    assert all(item.passed for item in checks(fixture))


def test_historical_local_wald_record_keeps_its_original_validation_contract():
    fixture = account()
    fixture[-1].identifications[0].obligations = fixture[-1].identifications[0].obligations[:2]
    assert all(item.passed for item in checks(fixture, version=14))
    assert any(not item.passed and item.rule == "causal_graph_consistency" for item in checks(fixture))


def test_changing_identification_label_cannot_bypass_typed_receipt_effect_obligations():
    fixture = account()
    fixture[-1].identifications[0].strategy = "other"
    fixture[-1].identifications[0].obligations = fixture[-1].identifications[0].obligations[:2]
    assert any(not item.passed and item.rule == "causal_graph_consistency" for item in checks(fixture))
