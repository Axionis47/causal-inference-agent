"""Provider role references hydrate only exact accepted bindings, retaining original bytes."""

import json
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
from uuid import uuid4

import pytest
from historical_analysis import work
from test_analysis_column_registry import fixture
from test_model_observability import client, response

from backend import model_access
from backend.analysis import domain, provider, references
from backend.analysis.contracts import Roles


@pytest.fixture(autouse=True)
def isolated():
    yield


def offered():
    context, ids, target, treatments, roles, _ = fixture()
    source = work.packet(context, {ids["c"]}, {"target": target.model_dump(), "treatments": treatments.model_dump()}, "Fixed question")
    source["established_variables"] = [roles.assignments[0].variable.model_dump(mode="json")]
    packet, aliases, choices = references.selections(source)
    choices["_variable_refs"] = [item["id"] for item in packet["established_variables"]]
    encoded = references.translate(roles.model_dump(mode="json"), {durable: alias for alias, durable in aliases.items()})
    encoded["assignments"][0]["variable"] = {"id": packet["established_variables"][0]["id"]}
    return context, ids, target, treatments, roles, source, packet, aliases, choices, encoded


def test_provider_role_variable_is_a_finite_pointer_while_public_contract_is_unchanged():
    *_, packet, aliases, choices, _ = offered()
    native = provider.provider_schema(Roles, choices)
    variable = native["properties"]["assignments"]["items"]["properties"]["variable"]
    assert variable["required"] == ["id"] and set(variable["properties"]) == {"id"}
    assert variable["properties"]["id"]["enum"] == [packet["established_variables"][0]["id"]]
    assert variable["properties"]["id"]["enum"][0] in aliases
    public = Roles.model_json_schema()["$defs"]["Variable"]
    assert {"label", "meaning", "column_ids", "derivation"} <= public["properties"].keys()


def test_decoder_restores_encoded_binding_before_reverse_aliases_and_preserves_other_claims():
    _, _, _, _, roles, _, packet, aliases, _, pointer = offered()
    original = deepcopy(pointer)
    registry = deepcopy(packet["established_variables"])
    decode = work.role_reference_decoder(registry)
    registry[0]["meaning"]["statement"] = "Caller mutation cannot replace the captured accepted binding."
    hydrated = decode(pointer)
    assert hydrated["assignments"][0]["variable"] == packet["established_variables"][0]
    assert references.translate(hydrated, aliases) == roles.model_dump(mode="json")
    assert pointer == original
    assert hydrated["assignments"][0]["account"] == pointer["assignments"][0]["account"]
    hydrated["assignments"][0]["variable"]["label"] = "Mutation"
    assert decode(pointer)["assignments"][0]["variable"] == packet["established_variables"][0]


@pytest.mark.parametrize("defect", ["unknown", "unoffered_durable", "extra", "copied_exact", "missing", "string", "nonlist", "nonobject"])
def test_decoder_rejects_unknown_or_overauthored_pointers_without_fuzzy_repair(defect):
    _, ids, _, _, _, _, packet, _, _, pointer = offered()
    if defect == "unknown":
        pointer["assignments"][0]["variable"]["id"] = "invented"
    elif defect == "unoffered_durable":
        pointer["assignments"][0]["variable"]["id"] = ids["c"]
    elif defect == "extra":
        pointer["assignments"][0]["variable"]["label"] = "Changed"
    elif defect == "copied_exact":
        pointer["assignments"][0]["variable"] = deepcopy(packet["established_variables"][0])
    elif defect == "missing":
        del pointer["assignments"][0]["variable"]
    elif defect == "string":
        pointer["assignments"][0]["variable"] = packet["established_variables"][0]["id"]
    elif defect == "nonlist":
        pointer["assignments"] = {}
    else:
        pointer["assignments"] = ["not an object"]
    with pytest.raises(ValueError):
        work.role_reference_decoder(packet["established_variables"])(pointer)


def test_empty_or_duplicate_registry_cannot_fall_back_to_variable_authorship():
    *_, packet, _, _, _ = offered()
    with pytest.raises(ValueError, match="without its accepted binding registry"):
        work.role_reference_decoder([])
    with pytest.raises(ValueError, match="unique"):
        work.role_reference_decoder(packet["established_variables"] * 2)
    with pytest.raises(ValueError, match="nonempty"):
        provider.provider_schema(Roles, {"_variable_refs": []})


def test_decoder_preserves_unknown_meaning_and_accepts_empty_widening_assignment_list():
    *_, packet, _, _, pointer = offered()
    packet["established_variables"][0]["meaning"] = {"statement": "Unknown measurement", "basis": "unknown", "references": [], "prerequisites": []}
    decode = work.role_reference_decoder(packet["established_variables"])
    assert decode(pointer)["assignments"][0]["variable"]["meaning"]["basis"] == "unknown"
    pointer["assignments"] = []
    pointer["requested_keys"], pointer["widening_reason"] = ["offered-key"], "Need an indexed timing statement."
    assert decode(pointer) == pointer
    del pointer["assignments"]
    assert decode(pointer) == pointer
    assert Roles.model_validate(decode(pointer)).assignments == []


def test_real_provider_boundary_retains_exact_pointer_text_and_telemetry(monkeypatch):
    *_, packet, _, choices, pointer = offered()
    raw = " \n" + json.dumps(pointer, indent=2) + "\n"
    calls = client(monkeypatch, response(raw))
    value = model_access.structured(provider.prepare(Roles, "Select exact references", packet, "test-model", choices=choices,
                                    reference_decoder=work.role_reference_decoder(packet["established_variables"])))
    assert value.assignments[0].variable.model_dump(mode="json") == packet["established_variables"][0]
    saved, metrics = model_access.response_artifacts(value)
    assert saved == raw and metrics["response_text_hash"] == sha256(raw.encode()).hexdigest()
    assert '"label"' not in raw and '"label"' in value.model_dump_json()
    assert len(calls) == 1


@pytest.mark.parametrize("defect", ["unknown_pointer", "extra_field", "invalid_json", "invalid_role"])
def test_decoder_or_validation_error_keeps_original_provider_observation(monkeypatch, defect):
    *_, packet, _, choices, pointer = offered()
    if defect == "unknown_pointer":
        pointer["assignments"][0]["variable"]["id"] = "invented"
    elif defect == "extra_field":
        pointer["assignments"][0]["variable"]["meaning"] = {"statement": "Must not overwrite accepted meaning"}
    elif defect == "invalid_role":
        pointer["assignments"][0]["roles"] = ["invented_role"]
    raw = '{"truncated":' if defect == "invalid_json" else json.dumps(pointer)
    calls = client(monkeypatch, response(raw, reason="MAX_TOKENS" if defect == "invalid_json" else "STOP"))
    with pytest.raises((model_access.ProviderResponseError, model_access.InvalidModelOutput)) as caught:
        model_access.structured(provider.prepare(Roles, "Select exact references", packet, "test-model", choices=choices,
                                reference_decoder=work.role_reference_decoder(packet["established_variables"])))
    saved, metrics = model_access.response_artifacts(caught.value)
    assert saved == raw and metrics["provider_response_id"] == "provider-response-123"
    assert metrics["response_text_hash"] == sha256(raw.encode()).hexdigest() and len(calls) == 1


def test_reference_schema_and_decoder_must_be_paired_before_provider_dispatch(monkeypatch):
    *_, packet, _, choices, pointer = offered()
    calls = client(monkeypatch, response(json.dumps(pointer)))
    with pytest.raises(ValueError, match="explicit accepted-registry decoder"):
        model_access.structured(provider.prepare(Roles, "Missing decoder", packet, "test-model", choices=choices))
    with pytest.raises(ValueError, match="vice versa"):
        model_access.structured(provider.prepare(Roles, "Missing finite schema", packet, "test-model", reference_decoder=work.role_reference_decoder(packet["established_variables"])))
    assert calls == []


def test_worker_charges_decoder_failure_then_corrects_with_same_exact_binding(monkeypatch):
    context, ids, target, treatments, _, source, packet, _, _, pointer = offered()
    scope = next(item for item in domain.role_scopes(context, target, treatments) if item["column_id"] == ids["c"])
    source.update(role_scope=scope, _model="test", _prompt="Assess roles", assumption_namespace="reference_fixture_")
    task = {"id": str(uuid4()), "kind": "role", "scope": {"role_scope": scope}, "column_ids": [ids["c"]], "packet": source, "attempts": []}
    invalid = deepcopy(pointer)
    invalid["assignments"][0]["variable"]["id"] = "unknown"
    raw = [json.dumps(invalid), "\n" + json.dumps(pointer)]
    reservations, calls, receipts = [], [], []

    class Client:
        def __init__(self, **kwargs):
            self.models = SimpleNamespace(generate_content=self.generate)

        def generate(self, **kwargs):
            calls.append(kwargs)
            return response(raw[len(calls) - 1])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def reserve(*args, **kwargs):
        reservations.append(deepcopy(kwargs))
        assert len(reservations) == len(calls) + 1
        return object()

    def finish(claimed, output, checks, **kwargs):
        receipt = {"id": "receipt:" + str(len(receipts)), "output": deepcopy(output), "checks": deepcopy(checks), **kwargs}
        receipts.append(receipt)
        task["attempts"].append(receipt)
        return receipt

    monkeypatch.setattr(model_access.genai, "Client", Client)
    monkeypatch.setattr(work.tasks, "view", lambda _: deepcopy(task))
    monkeypatch.setattr(work.tasks, "claim", reserve)
    monkeypatch.setattr(work.tasks, "finish", finish)
    work.run(task["id"])
    assert len(calls) == len(reservations) == 2
    assert [item["status"] for item in receipts] == ["rejected", "accepted"]
    assert [item["raw_output"] for item in receipts] == raw
    assert all(item["metrics"]["provider_response_id"] == "provider-response-123" for item in receipts)
    first, second = (item["request"] for item in reservations)
    assert first["payload"]["established_variables"] == second["payload"]["established_variables"] == packet["established_variables"]
    assert first["reference_decoding"] == second["reference_decoding"] == "accepted_variable_id"
    assert "correction" not in first["payload"] and "correction" in second["payload"]
    assert receipts[1]["output"]["assignments"][0]["variable"] == source["established_variables"][0]
