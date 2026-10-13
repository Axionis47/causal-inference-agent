"""Code gates for human context admission and nested frozen specifications."""

import pytest

from causal_agent.common.contracts import ContextRequirement, EndpointCatalogue, EndpointDefinition, Fact, FrozenDesign, assess_context


def catalogue(**requirement):
    return EndpointCatalogue(
        endpoints=[
            EndpointDefinition(
                id="example",
                title="example",
                targets=["ate"],
                trajectory=["freeze"],
                estimators=["linear"],
                requirements=[ContextRequirement(id="account", address="claim:assignment.kind", **requirement)],
            )
        ]
    )


def admit(cat, facts, **kwargs):
    return assess_context(cat, facts, family="example", memory_version=2, data_version=1, **kwargs)


def test_a_model_draft_does_not_admit_context_even_when_it_claims_confirmation():
    facts = {"claim:assignment.kind": Fact(value="random", status="confirmed", source="model:infer")}
    assert not admit(catalogue(), facts).ready
    facts["claim:assignment.kind"].source = None
    assert not admit(catalogue(), facts).ready
    facts["claim:assignment.kind"].source = "user:turn:2"
    assert admit(catalogue(), facts).ready


def test_unknown_requires_explicit_endpoint_policy_and_recorded_human_source():
    facts = {"claim:assignment.kind": Fact(status="unknown", source="user:turn:1")}
    assert not admit(catalogue(), facts).ready
    accepted = admit(catalogue(allow_unknown=True), facts)
    assert accepted.ready and accepted.endpoints[0].status == "conditional"
    facts["claim:assignment.kind"].source = "model:infer"
    assert not admit(catalogue(allow_unknown=True), facts).ready


def test_modifier_requirements_expand_every_named_group_column():
    cat = catalogue()
    cat.endpoints[0].requirements = [ContextRequirement(id="baseline", address="col:<modifier>.when", values=["before"])]
    facts = {
        "col:a.when": Fact(value="before", status="confirmed", source="user:turn:1"),
        "col:b.when": Fact(value="after", status="confirmed", source="user:turn:1"),
    }
    result = admit(cat, facts, bindings={"modifier": ["a", "b"]})
    assert result.endpoints[0].status == "excluded"
    assert len(result.endpoints[0].obligations) == 2
    assert admit(cat, facts, bindings={"modifier": []}).missing == ["claim:target.modifiers"]


def test_nested_changes_break_the_freeze_seal_and_cannot_be_reloaded():
    class Design(FrozenDesign):
        parameters: dict

    design = Design(parameters={"horizons": [0, 1]})
    design.assert_integrity()
    design.parameters["horizons"].pop()
    with pytest.raises(ValueError, match="changed"):
        design.assert_integrity()
    with pytest.raises(ValueError, match="integrity"):
        Design.model_validate(design.model_dump())
