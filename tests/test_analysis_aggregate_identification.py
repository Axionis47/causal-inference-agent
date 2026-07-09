"""Weighted identification closes over existing component criteria without new premises."""

from copy import deepcopy

import pytest
from test_analysis_contrasts import aggregate, binding
from test_scientific_v8 import claim, decision

from backend.analysis import domain
from backend.analysis.contracts import CausalAccount, IdentificationClaim


@pytest.fixture(autouse=True)
def isolated():
    yield


def fixture():
    context, ids, target, treatments, roles = aggregate()
    treatments = domain.resolve(treatments, {})
    identities = [effect.id for effect in treatments.effects]
    for role in roles.assignments:
        role.roles = ["irrelevant"]
        role.effect_ids = identities.copy()
        role.operational_use = []
        if role.variable.id == ids["b"]:
            role.roles, role.operational_use = ["identifier"], ["unit_key"]
        elif role.variable.id == ids["time"]:
            role.roles, role.operational_use = ["time"], ["time_key"]
    proposal = CausalAccount(**decision(), nodes=[{"id": identity} for identity in ids.values()],
        edges=[{"source": ids["t"], "target": ids["y"], "basis": "supplied", "references": ["question"], "explanation": "Declared fixture pathway."}],
        identification="Component DiD effects and their explicitly requested fixed average.",
        identifications=[
            {"effect_id": identity, "strategy": "aggregate" if identity == "average" else "did", "evidence": claim(),
             "obligations": [] if identity == "average" else [{"name": name, "claim": claim()} for name in ("parallel_trends", "no_anticipation", "comparison_support")]}
            for identity in reversed(identities)
        ])
    proposal = domain.canonical_graph(proposal, target, treatments, roles)
    prior = {"target": target.model_dump(), "treatments": treatments.model_dump(), "roles": roles.model_dump()}
    return context, ids, proposal, prior


def checks(proposal, context, prior):
    return domain.checks(proposal, context, prior, scientific_version=15)


def failure(proposal, context, prior, text):
    result = checks(proposal, context, prior)
    assert any(not gate.passed and text in gate.detail for gate in result), result
    return result


def test_did_average_validates_components_first_and_exposes_frozen_weights_without_mutation():
    context, _, proposal, prior = fixture()
    raw = proposal.model_dump()
    frozen = deepcopy(prior)
    result = checks(proposal, context, prior)
    assert all(gate.passed for gate in result), result
    gate = next(gate for gate in result if gate.rule == "causal_graph_consistency")
    assert "primary (weight 0.5), later (weight 0.5)" in gate.detail
    assert "conditional premises" in gate.detail and "not source entailment" in gate.detail
    assert proposal.model_dump() == raw and prior == frozen
    assert proposal.identifications[0].effect_id == "average", "Validation order is independent of proposal order."
    assert proposal.identifications[0].obligations == []


def test_nested_averages_follow_topological_component_order():
    context, _, proposal, prior = fixture()
    prior["treatments"]["effects"][0]["comparison"] = binding("aggregate", terms=[{"effect_id": "later", "weight": 1}]).model_dump()
    first = next(item for item in proposal.identifications if item.effect_id == "primary")
    first.strategy, first.obligations = "aggregate", []
    result = checks(proposal, context, prior)
    assert all(gate.passed for gate in result), result
    detail = next(gate.detail for gate in result if gate.rule == "causal_graph_consistency")
    assert detail.index("primary: later (weight 1.0)") < detail.index("average: primary")


@pytest.mark.parametrize("defect,message", [
    ("missing_claim", "exactly one explicit identification"),
    ("missing_obligation", "parallel_trends"),
    ("unknown_evidence", "supplied evidence or a declared"),
    ("unknown_obligation", "distinct and supported"),
    ("unidentified_component", "explicit conditional obligations"),
    ("invalid_graph_criterion", "backdoor criterion"),
])
def test_aggregate_cannot_hide_a_missing_or_invalid_component(defect, message):
    context, ids, proposal, prior = fixture()
    component = next(item for item in proposal.identifications if item.effect_id == "primary")
    if defect == "missing_claim":
        proposal.identifications.remove(component)
    elif defect == "missing_obligation":
        component.obligations.pop(0)
    elif defect == "unknown_evidence":
        component.evidence.basis, component.evidence.references = "unknown", []
    elif defect == "unknown_obligation":
        component.obligations[0].claim.basis, component.obligations[0].claim.references = "unknown", []
    elif defect == "unidentified_component":
        component.strategy, component.obligations = "other", []
    else:
        component.strategy = "backdoor"
        edge = proposal.edges[0].model_copy(deep=True)
        edge.source, edge.target = ids["score"], ids["t"]
        proposal.edges.append(edge)
        edge = edge.model_copy(deep=True)
        edge.target = ids["y"]
        proposal.edges.append(edge)
    result = failure(proposal, context, prior, message)
    assert not next(gate for gate in result if gate.rule == "causal_graph_consistency").passed


@pytest.mark.parametrize("defect,message", [
    ("ordinary_effect", "exact accepted typed aggregate"),
    ("missing_payload", "exact accepted typed aggregate"),
    ("other_strategy", "requires aggregate identification"),
    ("new_adjustment", "cannot add adjustment"),
    ("new_obligation", "cannot add adjustment"),
    ("unknown_component", "distinct existing component"),
    ("duplicate_component", "distinct existing component"),
    ("self", "cannot depend on itself"),
    ("cycle", "cycle"),
    ("scope", "component treatment scope"),
    ("weight", "weights to sum to one"),
    ("nonfinite_weight", "weights to sum to one"),
])
def test_aggregate_has_no_generic_certificate_or_new_scientific_choice(defect, message):
    context, ids, proposal, prior = fixture()
    average = proposal.identifications[0]
    comparison = prior["treatments"]["effects"][-1]["comparison"]
    if defect == "ordinary_effect":
        proposal.identifications[1].strategy = "aggregate"
    elif defect == "missing_payload":
        comparison["aggregate"] = None
    elif defect == "other_strategy":
        average.strategy = "did"
    elif defect == "new_adjustment":
        # Keep an eligible scope so rejection tests the aggregate-specific rule.
        role = next(item for item in prior["roles"]["assignments"] if item["variable"]["id"] == ids["score"])
        role["roles"] = ["confounder"]
        node = next(item for item in proposal.nodes if item.id == ids["score"])
        node.roles = ["confounder"]
        average.adjustment = [ids["score"]]
    elif defect == "new_obligation":
        average.obligations = [proposal.identifications[1].obligations[0].model_copy(deep=True)]
    elif defect == "unknown_component":
        comparison["aggregate"]["terms"][0]["effect_id"] = "invented_auxiliary"
    elif defect == "duplicate_component":
        comparison["aggregate"]["terms"][0]["effect_id"] = "later"
    elif defect == "self":
        comparison["aggregate"]["terms"][0]["effect_id"] = "average"
    elif defect == "cycle":
        prior["treatments"]["effects"][0]["comparison"] = binding("aggregate", terms=[{"effect_id": "average", "weight": 1}]).model_dump()
        component = next(item for item in proposal.identifications if item.effect_id == "primary")
        component.strategy, component.obligations = "aggregate", []
    elif defect == "scope":
        prior["treatments"]["effects"][-1]["treatment_ids"].append(ids["b"])
    elif defect == "weight":
        comparison["aggregate"]["terms"][0]["weight"] = .4
    else:
        comparison["aggregate"]["terms"][0]["weight"] = float("inf")
    failure(proposal, context, prior, message)


def test_unoffered_component_provenance_still_blocks_the_whole_checkpoint():
    context, _, proposal, prior = fixture()
    proposal.identifications[1].obligations[0].claim.references = ["columns.unoffered.definition.invented"]
    failure(proposal, context, prior, "References outside")


def test_aggregate_strategy_is_not_a_historical_untyped_escape_hatch():
    context, _, proposal, prior = fixture()
    result = domain.checks(proposal, context, prior, scientific_version=9)
    assert any(not gate.passed and "exact accepted typed aggregate" in gate.detail for gate in result)


def test_aggregate_contract_has_no_second_component_identity_or_weight_source():
    assert set(IdentificationClaim.model_fields) == {"effect_id", "strategy", "adjustment", "evidence", "obligations"}
