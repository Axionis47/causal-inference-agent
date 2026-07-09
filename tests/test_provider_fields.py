"""Wire compression must preserve literals, typed contracts and authority checks."""

import json
from copy import deepcopy

import pytest
from test_analysis_contrasts import fixture
from test_model_observability import client, response

from backend import model_access
from backend.analysis import contracts, provider, provider_fields


@pytest.fixture(autouse=True)
def isolated():
    yield


SCHEMA = {
    "type": "object", "required": ["source_reference", "operands"],
    "properties": {
        "source_reference": {"type": "string", "enum": ["p0"]},
        "operands": {"type": "array", "items": {"anyOf": [{"$ref": "#/$defs/Value"}, {"type": "null"}]}},
    },
    "$defs": {"Value": {"type": "object", "required": ["literal", "weight"], "properties": {
        "literal": {"type": "string"}, "weight": {"type": "number"},
    }}},
}


def test_nested_compact_response_preserves_literal_text_and_exact_references():
    original = deepcopy(SCHEMA)
    wire = {"p0": "p0", "p1": [{"p0": "source_reference = p1, μ", "p1": -1}, None]}
    assert provider_fields.expand(wire, SCHEMA) == {
        "source_reference": "p0", "operands": [{"literal": "source_reference = p1, μ", "weight": -1}, None],
    }
    compact = provider_fields.compact(SCHEMA)
    assert compact["properties"]["p0"]["enum"] == ["p0"]
    assert compact["required"] == ["p0", "p1"]
    assert compact["$defs"]["Value"]["required"] == ["p0", "p1"]
    assert compact["$defs"]["Value"]["properties"]["p1"]["description"].startswith("weight.")
    assert SCHEMA == original


@pytest.mark.parametrize("wire", [
    {"source_reference": "p0", "p1": []},
    {"p0": "p0", "p1": [], "p2": "invented"},
    {"p0": "p0"},
    {"p0": "p0", "p1": [{"p0": "x", "weight": 1}]},
    {"p0": "p0", "p1": [{"p0": "x"}]},
    {"p0": "p0", "p1": "not an array"},
])
def test_compact_decoder_rejects_unknown_missing_and_wrongly_shaped_fields(wire):
    with pytest.raises(ValueError):
        provider_fields.expand(wire, SCHEMA)


def wire_value(value, schema, root=None):
    """Emulate provider field selection for full domain validation tests."""
    root = root or schema
    if "$ref" in schema:
        return wire_value(value, root["$defs"][schema["$ref"].split("/")[-1]], root)
    if "anyOf" in schema:
        if value is None:
            return None
        return wire_value(value, next(s for s in schema["anyOf"] if s.get("type") != "null"), root)
    if schema.get("type") == "object":
        return {f"p{i}": wire_value(value.get(name), child, root) for i, (name, child) in enumerate(schema["properties"].items())}
    if schema.get("type") == "array":
        return [wire_value(item, schema["items"], root) for item in value]
    return value


def test_compact_transport_preserves_full_treatment_contract_and_raw_observation(monkeypatch):
    _, _, _, proposal, _ = fixture()
    choices = {"_json_schema": ["treatments"], "_explicit_fields": ["treatments"]}
    canonical = provider.provider_schema(contracts.Treatments, choices)
    raw = json.dumps(wire_value(proposal.model_dump(mode="json"), canonical))
    calls = client(monkeypatch, response(raw))
    result = model_access.structured(provider.prepare(
        contracts.Treatments, "Test exact typed restoration", {}, "test",
        choices={**choices, "_compact_fields": ["treatments"]},
    ))
    assert result == proposal
    assert model_access.response_artifacts(result)[0] == raw
    assert calls[0]["config"].response_json_schema == provider_fields.compact(canonical)


def test_compact_fields_do_not_admit_invalid_typed_values(monkeypatch):
    _, _, _, proposal, _ = fixture()
    choices = {"_json_schema": ["treatments"], "_explicit_fields": ["treatments"]}
    canonical = provider.provider_schema(contracts.Treatments, choices)
    value = proposal.model_dump(mode="json")
    value["values"][0]["value"]["kind"] = "invented"
    client(monkeypatch, response(json.dumps(wire_value(value, canonical))))
    with pytest.raises(model_access.ProviderResponseError, match="canonical choices"):
        model_access.structured(provider.prepare(contracts.Treatments, "Test", {}, "test",
            choices={**choices, "_compact_fields": ["treatments"]}))


def test_large_reference_enums_are_enforced_locally_before_claim_admission():
    schema = {"type": "object", "required": ["references"], "properties": {
        "references": {"type": "array", "items": {"type": "string", "enum": ["exact-evidence"]}},
    }}
    assert "enum" not in provider_fields.compact(schema)["properties"]["p0"]["items"]
    assert provider_fields.expand({"p0": ["exact-evidence"]}, schema) == {"references": ["exact-evidence"]}
    with pytest.raises(ValueError, match="canonical choices"):
        provider_fields.expand({"p0": ["invented-evidence"]}, schema)
