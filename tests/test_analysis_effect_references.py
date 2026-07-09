"""Accepted effect questions are pointers, while new comparisons retain their own gates."""

import json
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
from uuid import UUID

import pytest
from historical_analysis import checkpoints, work
from test_analysis_contrasts import fixture
from test_model_observability import client, response
from test_provider_fields import wire_value

from backend import model_access
from backend.analysis import domain, effect_references, provider, references
from backend.analysis.contracts import RequestedEffect, Treatments
from backend.context.contracts import DatasetStatement, SourceReference


@pytest.fixture(autouse=True)
def isolated():
    yield


def offered():
    context, ids, target, treatments, _ = fixture()
    source = SourceReference(kind="message", id=UUID(int=1), author_id=UUID(int=2))
    context.dataset.study_process["requested"] = DatasetStatement(value="The supplied requested contrast.", source=source)
    target.requested_effects[0].question.references.append("dataset.study_process.requested")
    target.requested_effects.append(RequestedEffect(
        id="secondary", question={"statement": "The second exact requested question.", "basis": "supplied", "references": ["question"]},
        priority="secondary",
    ))
    secondary = treatments.effects[0].model_copy(deep=True)
    secondary.id, secondary.priority = "secondary", "secondary"
    treatments.effects.append(secondary)
    source_packet = work.packet(context, set(ids.values()), {"target": target.model_dump(mode="json")}, "Fixed question")
    packet, aliases, choices = references.selections(source_packet)
    choices.update(_effect_refs=[item["id"] for item in packet["decisions"]["target"]["requested_effects"]],
                   _explicit_fields=["treatments"], _json_schema=["treatments"])
    pointer = references.translate(treatments.model_dump(mode="json"), {durable: alias for alias, durable in aliases.items()})
    for item in pointer["effects"]:
        del item["contrast"], item["priority"]
    return context, target, source_packet, packet, aliases, choices, pointer


def decode(packet):
    return effect_references.decoder(packet["decisions"]["target"], packet["decision_refs"])


def test_finite_effect_schema_removes_only_duplicate_fields_without_changing_public_contract():
    *_, choices, _ = offered()
    public = deepcopy(Treatments.model_json_schema())
    compact = provider.provider_schema(Treatments, choices)
    effect = compact["$defs"]["EffectRequest"]
    assert set(effect["properties"]) == set(effect["required"]) == {"id", "treatment_ids", "estimand", "comparison"}
    assert effect["properties"]["id"]["enum"] == ["primary", "secondary"]
    assert "evidence" in compact["$defs"]["ContrastBinding"]["properties"]
    assert {"contrast", "priority"} <= public["$defs"]["EffectRequest"]["properties"].keys()
    assert Treatments.model_json_schema() == public
    choices.pop("_json_schema")
    assert provider.provider_schema(Treatments, choices)["properties"]["effects"]["items"]["properties"]["id"]["enum"] == ["primary", "secondary"]


def test_hydration_preserves_claim_authority_then_reverses_aliases_without_changing_new_comparison():
    _, target, _, packet, aliases, _, pointer = offered()
    original = deepcopy(pointer)
    hydrate = decode(packet)
    packet["decisions"]["target"]["requested_effects"][0]["question"]["statement"] = "Caller mutation"
    hydrated = hydrate(pointer)
    restored = references.translate(hydrated, aliases)
    for item, requested in zip(restored["effects"], target.requested_effects, strict=True):
        expected = requested.question.model_dump(mode="json")
        expected["prerequisites"] = ["decision:target"]
        assert item["contrast"] == expected and item["priority"] == requested.priority
    assert hydrated["effects"][0]["comparison"] == pointer["effects"][0]["comparison"]
    assert pointer == original
    hydrated["effects"][0]["contrast"]["statement"] = "Mutation"
    assert hydrate(pointer)["effects"][0]["contrast"]["statement"] != "Mutation"


@pytest.mark.parametrize("defect", ["unknown", "duplicate", "missing_ready", "missing_id", "reauthored_contrast", "reauthored_priority", "extra", "nonlist", "nonobject"])
def test_reference_decoder_rejects_invalid_or_overauthored_selections(defect):
    *_, packet, _, _, pointer = offered()
    if defect == "unknown":
        pointer["effects"][0]["id"] = "invented"
    elif defect == "duplicate":
        pointer["effects"][1]["id"] = "primary"
    elif defect == "missing_ready":
        pointer["effects"].pop()
    elif defect == "missing_id":
        del pointer["effects"][0]["id"]
    elif defect == "reauthored_contrast":
        pointer["effects"][0]["contrast"] = deepcopy(packet["decisions"]["target"]["requested_effects"][0]["question"])
    elif defect == "reauthored_priority":
        pointer["effects"][0]["priority"] = "primary"
    elif defect == "extra":
        pointer["effects"][0]["question"] = "New prose"
    elif defect == "nonlist":
        pointer["effects"] = {}
    else:
        pointer["effects"] = ["primary"]
    with pytest.raises(ValueError):
        decode(packet)(pointer)


@pytest.mark.parametrize("defect", ["missing_prerequisite", "not_ready", "empty", "duplicate", "no_primary", "unknown_question"])
def test_decoder_requires_the_exact_ready_target_registry(defect):
    *_, packet, _, _, _ = offered()
    target = packet["decisions"]["target"]
    if defect == "missing_prerequisite":
        packet["decision_refs"] = []
    elif defect == "not_ready":
        target["status"] = "needs_input"
    elif defect == "empty":
        target["requested_effects"] = []
    elif defect == "duplicate":
        target["requested_effects"] *= 2
    elif defect == "no_primary":
        target["requested_effects"][0]["priority"] = "secondary"
    else:
        target["requested_effects"][0]["question"]["basis"] = "unknown"
    with pytest.raises(ValueError):
        decode(packet)


def test_nonready_partial_effects_remain_partial_and_cannot_invent_an_effect():
    *_, packet, _, _, pointer = offered()
    pointer.update(status="needs_input", unresolved=["Missing treatment meaning."], effects=[])
    assert decode(packet)(pointer) == pointer
    del pointer["effects"]
    assert decode(packet)(pointer) == pointer


@pytest.mark.parametrize("defect", [None, "decision_source", "invalid_weights", "unknown_source"])
def test_carry_forward_does_not_repair_or_approve_independent_typed_comparison_claims(defect):
    context, target, _, packet, aliases, _, pointer = offered()
    comparison = pointer["effects"][0]["comparison"]
    if defect == "decision_source":
        comparison["evidence"]["references"] = ["decision:target"]
    elif defect == "invalid_weights":
        comparison["static"]["terms"][0]["weight"] = 2
    elif defect == "unknown_source":
        comparison["evidence"]["references"] = ["invented.source"]
    hydrated = decode(packet)(pointer)
    assert hydrated["effects"][0]["comparison"] == comparison
    value = domain.resolve(Treatments.model_validate(references.translate(hydrated, aliases)), {})
    checks = domain.checks(value, context, {"target": target.model_dump()}, scientific_version=15)
    assert all(item.passed for item in checks) is (defect is None), checks


def test_shared_provider_boundary_saves_pointer_bytes_request_and_hydrated_proposal(monkeypatch):
    _, target, source, packet, _, choices, pointer = offered()
    raw = " \n" + json.dumps(wire_value(pointer, provider.provider_schema(Treatments, choices)), indent=2) + "\n"
    calls = client(monkeypatch, response(raw))
    h = checkpoints.Harness.__new__(checkpoints.Harness)
    h.saved, h.db, h.shared_owner = {}, object(), None
    h.run = SimpleNamespace(id=UUID(int=1), calls=1, stage="treatments", inputs={"model": "test"})
    h.event = lambda *args, **kwargs: None
    reservations, settlements = [], []

    def reserve(db, run, key, input_hash, metrics, **kwargs):
        reservations.append(deepcopy(metrics))
        return key

    monkeypatch.setattr(checkpoints.tasks, "reserve_shared", reserve)
    monkeypatch.setattr(checkpoints.tasks, "settle_shared", lambda *args, **kwargs: settlements.append(kwargs))
    value = h.ask("treatments:0", Treatments, "Bind exact effects", source)
    assert len(calls) == len(reservations) == len(settlements) == 1
    assert settlements[0]["raw_output"] == raw and settlements[0]["metrics"]["response_text_hash"] == sha256(raw.encode()).hexdigest()
    assert settlements[0]["receipt"] == value.model_dump(mode="json")
    assert reservations[0]["reference_decoding"] == reservations[0]["request"]["reference_decoding"] == "accepted_target_effect_id"
    assert reservations[0]["request"]["payload"] == {key: value for key, value in packet.items() if not key.startswith("_")}
    assert reservations[0]["request"]["aliases"]
    assert value.effects[0].contrast.statement == target.requested_effects[0].question.statement
    assert value.effects[0].contrast.references == target.requested_effects[0].question.references
    assert value.effects[0].contrast.prerequisites == ["decision:target"]
    assert '"contrast"' not in raw and '"priority"' not in raw


@pytest.mark.parametrize("defect", ["unknown", "reauthored", "invalid_json"])
def test_decoder_failure_retains_exact_raw_response_without_fallback(monkeypatch, defect):
    *_, packet, _, choices, pointer = offered()
    if defect == "unknown":
        pointer["effects"][0]["id"] = "unknown"
    elif defect == "reauthored":
        pointer["effects"][0]["contrast"] = {"statement": "Do not silently repair this", "basis": "supplied", "references": ["decision:target"]}
    raw = '{"truncated":' if defect == "invalid_json" else json.dumps(pointer)
    calls = client(monkeypatch, response(raw))
    with pytest.raises((model_access.ProviderResponseError, model_access.InvalidModelOutput)) as caught:
        model_access.structured(provider.prepare(Treatments, "Bind exact effects", packet, "test", choices=choices, reference_decoder=decode(packet)))
    saved, metrics = model_access.response_artifacts(caught.value)
    assert saved == raw and metrics["response_text_hash"] == sha256(raw.encode()).hexdigest()
    assert len(calls) == 1


def test_effect_reference_schema_requires_its_decoder_before_dispatch(monkeypatch):
    *_, packet, _, choices, pointer = offered()
    calls = client(monkeypatch, response(json.dumps(pointer)))
    with pytest.raises(ValueError, match="explicit accepted-registry decoder"):
        model_access.structured(provider.prepare(Treatments, "Bind exact effects", packet, "test", choices=choices))
    with pytest.raises(ValueError, match="vice versa"):
        model_access.structured(provider.prepare(Treatments, "Bind exact effects", packet, "test", reference_decoder=decode(packet)))
    with pytest.raises(ValueError, match="nonempty"):
        provider.provider_schema(Treatments, {"_effect_refs": []})
    assert calls == []
