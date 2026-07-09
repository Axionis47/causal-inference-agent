"""Compact provider encoding preserves scientific contracts and exact attempt records."""

import json
from copy import deepcopy
from types import SimpleNamespace
from uuid import UUID

import pytest
from google.genai import _common, models, types
from historical_analysis import checkpoints
from model_double import model_double
from pydantic import ValidationError
from test_analysis_contrasts import accepted, fixture, rejected
from test_model_observability import client, response

from backend import model_access
from backend.analysis import contracts, provider


@pytest.fixture(autouse=True)
def isolated():
    yield


def without_required(value):
    if isinstance(value, dict):
        return {key: without_required(item) for key, item in value.items() if key != "required"}
    if isinstance(value, list):
        return [without_required(item) for item in value]
    return value


def nodes(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from nodes(item)


def target_dependency():
    target = contracts.Target(
        status="ready", summary="The requested effect is accepted.", changes="Bound request.",
        permits="Bind treatment levels.", requested_effects=[contracts.RequestedEffect(
            id="requested_effect", priority="primary",
            question=contracts.Claim(basis="supplied", statement="The exact requested comparison.", references=["question"]),
        )],
    )
    return {"decisions": {"target": target.model_dump(mode="json")}, "decision_refs": ["decision:target"]}


def test_explicit_treatment_encoding_only_changes_native_required_lists():
    public = deepcopy(contracts.Treatments.model_json_schema())
    choices = {"column_ids": ["c000000000001", "c000000000002"], "effect_id": ["primary"],
               "citations": ["question"], "requested_keys": [], "prerequisites": ["decision:target"]}
    original_choices = deepcopy(choices)
    native = provider.provider_schema(contracts.Treatments, choices)
    explicit = provider.provider_schema(contracts.Treatments, {**choices, "_explicit_fields": ["treatments"]})

    assert native != explicit
    assert without_required(native) == without_required(explicit)
    objects = [node for node in nodes(explicit) if node.get("type") == "object" and "properties" in node]
    assert len(objects) > 25  # Includes every nested contrast and assignment branch.
    assert all(node["required"] == list(node["properties"]) for node in objects)
    assert all("required" not in node for node in nodes(explicit) if node.get("type") != "object")
    assert contracts.Treatments.model_json_schema() == public
    assert choices == original_choices
    assert provider.provider_schema(contracts.Treatments, choices) == native
    # A property named "properties" is a list of Claims, not an object-schema map.
    types.GenerateContentConfig(response_schema=explicit).model_dump(mode="json")


def test_explicit_presence_does_not_remove_nullable_alternatives_or_infer_defaults():
    native = provider.provider_schema(contracts.Treatments, {"_explicit_fields": ["treatments"]})
    comparison = native["properties"]["effects"]["items"]["properties"]["comparison"]
    assert {"type": "null"} in comparison["anyOf"]
    payload = comparison["anyOf"][0]
    for name in ("static", "group_time", "discontinuity", "aggregate", "unsupported"):
        assert name in payload["required"]
        assert {"type": "null"} in payload["properties"][name]["anyOf"]
    assert all("default" not in node for node in nodes(native))
    historical = contracts.Treatments(status="needs_input", summary="No accepted treatment yet.",
                                      changes="Awaiting source meaning.", permits="No execution.")
    assert historical.values == [] and historical.effects == []


@pytest.mark.parametrize("defect", ["unknown_reference", "extra_field", "invalid_weight"])
def test_explicit_encoding_keeps_local_treatment_validation_authoritative(defect):
    context, _, target, proposal, _ = fixture()
    provider.provider_schema(contracts.Treatments, {"_explicit_fields": ["treatments"]})
    accepted(proposal, context, target)
    if defect == "extra_field":
        raw = proposal.model_dump(mode="json")
        raw["values"][0]["guessed_level"] = "one"
        with pytest.raises(ValidationError, match="Extra inputs"):
            contracts.Treatments.model_validate(raw)
    elif defect == "unknown_reference":
        proposal.values[0].meaning.references = ["invented.source"]
        rejected(proposal, context, target, "References outside")
    else:
        proposal.effects[0].comparison.static.terms[0].weight = 2
        rejected(proposal, context, target, "sum")


@pytest.mark.parametrize("schema", [contracts.Treatments, contracts.Target, contracts.Roles,
                                  contracts.CausalAccount, contracts.DiagnosticAction, contracts.Plan,
                                  contracts.Program, contracts.CodeReview, contracts.Conclusion])
def test_shared_harness_pins_explicit_encoding_only_for_treatments(monkeypatch, schema):
    h = checkpoints.Harness.__new__(checkpoints.Harness)
    h.saved, h.db, h.shared_owner = {}, object(), None
    h.run = SimpleNamespace(id=UUID(int=1), calls=1, stage="semantic", inputs={"model": "test"})
    h.event = lambda *args, **kwargs: None
    reservations, calls = [], []
    payload = {"question": "Only inspect native request construction.", "_private": "Not offered."}
    if schema is contracts.Treatments:
        payload.update(target_dependency())
    original = deepcopy(payload)

    def reserve(db, run, key, input_hash, metrics, **kwargs):
        reservations.append(deepcopy(metrics["request"]))
        return key

    def inspect(given_schema, prompt, offered, model, **kwargs):
        assert reservations, "The exact request must be persisted before provider dispatch."
        calls.append(kwargs["prepared"].schema)
        assert reservations[0]["schema"] == kwargs["prepared"].schema
        assert offered == {key: value for key, value in payload.items() if not key.startswith("_")}
        assert callable(kwargs.get("reference_decoder")) == (schema is contracts.Treatments)
        raise RuntimeError("Stop before provider access.")

    monkeypatch.setattr(checkpoints.tasks, "reserve_shared", reserve)
    monkeypatch.setattr(checkpoints.tasks, "settle_shared", lambda *args, **kwargs: None)
    monkeypatch.setattr(model_access, "structured", model_double(inspect))
    with pytest.raises(RuntimeError, match="Stop before provider access"):
        h.ask("synthetic:0", schema, "Inspect encoding", payload)
    expected = {"_explicit_fields": ["treatments"], "_json_schema": ["treatments"], "_compact_fields": ["treatments"], "_effect_refs": ["requested_effect"]} if schema is contracts.Treatments else {}
    assert calls == [provider.provider_schema(schema, expected)]
    assert reservations[0]["schema_format"] == ("response_json_schema" if schema is contracts.Treatments else "response_schema")
    assert payload == original
    if schema is not contracts.Treatments:
        assert reservations[0]["schema"] == provider.provider_schema(schema)


def expand(schema):
    def visit(value, ancestors=()):
        if isinstance(value, dict):
            if "$ref" in value:
                assert set(value) == {"$ref"}, "JSON Schema references must stand alone."
                ref = value["$ref"]
                assert ref.startswith("#/$defs/") and ref not in ancestors
                return visit(schema["$defs"][ref.removeprefix("#/$defs/")], (*ancestors, ref))
            return {key: visit(item, ancestors) for key, item in value.items() if key != "$defs"}
        if isinstance(value, list):
            return [visit(item, ancestors) for item in value]
        return value
    return visit(schema)


@pytest.mark.parametrize("columns,effects", [(3, 2), (5, 2), (4, 4)])
def test_shared_definitions_expand_to_exact_native_treatments_constraints(columns, effects):
    public = deepcopy(contracts.Treatments.model_json_schema())
    choices = {"column_ids": [f"c{i:012x}" for i in range(columns)],
               "effect_id": [f"requested_effect_{i}" for i in range(effects)],
               "citations": ["question", *[f"e{i:012x}" for i in range(20)]],
               "requested_keys": [], "prerequisites": ["decision:target"],
               "_assumption_namespace": ["treatments_"], "_explicit_fields": ["treatments"]}
    native = provider.provider_schema(contracts.Treatments, choices)
    compact = provider.provider_schema(contracts.Treatments, {**choices, "_json_schema": ["treatments"]})
    assert set(compact["$defs"]) == set(public["$defs"])
    assert expand(compact) == native
    assert len(json.dumps(compact)) < .7 * len(json.dumps(native))
    assert contracts.Treatments.model_json_schema() == public


def test_compact_transport_uses_only_json_schema_and_keeps_exact_raw_response(monkeypatch):
    proposal = contracts.Treatments(status="needs_input", summary="No study supplied.",
                                    changes="No decision.", permits="No execution.")
    raw = " \n" + proposal.model_dump_json() + "\n"
    calls = client(monkeypatch, response(raw))
    choices = {"_explicit_fields": ["treatments"], "_json_schema": ["treatments"]}
    value = model_access.structured(provider.prepare(contracts.Treatments, "No study supplied.", {}, "gemini-2.5-flash",
                                    choices=choices, max_output_tokens=8192, thinking_budget=2048))
    assert len(calls) == 1 and value == proposal
    config = calls[0]["config"]
    assert config.response_schema is None
    assert config.response_json_schema == provider.provider_schema(contracts.Treatments, choices)
    wire = _common.convert_to_dict(models._GenerateContentConfig_to_vertex(SimpleNamespace(vertexai=True), config))
    assert "responseSchema" not in wire and wire["responseJsonSchema"] == config.response_json_schema
    assert wire["responseMimeType"] == "application/json" and wire["maxOutputTokens"] == 8192
    assert wire["thinkingConfig"]["thinking_budget"] == 2048
    saved, metrics = model_access.response_artifacts(value)
    assert saved == raw and metrics["provider_schema_format"] == "response_json_schema"


def test_compact_transport_still_rejects_invalid_local_output_without_fallback(monkeypatch):
    raw = '{"status":"needs_input","summary":"Missing required local decision fields"}'
    calls = client(monkeypatch, response(raw))
    with pytest.raises(model_access.InvalidModelOutput) as caught:
        model_access.structured(provider.prepare(contracts.Treatments, "Test only", {}, "test",
                                choices={"_json_schema": ["treatments"], "_explicit_fields": ["treatments"]}))
    saved, metrics = model_access.response_artifacts(caught.value)
    assert saved == raw and metrics["provider_schema_format"] == "response_json_schema"
    assert len(calls) == 1 and calls[0]["config"].response_schema is None


@pytest.mark.parametrize("traced", [False, True])
def test_shared_failure_recovery_reserves_parent_trace_and_exact_event_fence(monkeypatch, traced):
    h = checkpoints.Harness.__new__(checkpoints.Harness)
    h.saved, h.db, h.shared_owner = {}, object(), object()
    h.run = SimpleNamespace(id=UUID(int=1), calls=12, stage="treatments", inputs={"model": "test"})
    reservations, events, settlements = [], [], []
    h.event = lambda *args, **kwargs: events.append(args)
    trees = [SimpleNamespace(id=UUID(int=i), session_name="project", get_url=lambda i=i: f"https://trace.test/{i}") for i in (2, 3)]
    current = iter(trees if traced else [None, None])
    monkeypatch.setattr(checkpoints, "get_current_run_tree", lambda: next(current))

    def reserve(db, run, key, input_hash, metrics, **kwargs):
        assert kwargs["recover"] and kwargs["owner"] is h.shared_owner
        run.calls += 1
        reservation = key if not reservations else key + ":transport:1"
        reservations.append((reservation, deepcopy(metrics)))
        return reservation

    def fail(*args, **kwargs):
        assert events[-1][4]["provider_attempt_key"] == "shared:" + reservations[-1][0]
        assert events[-1][0] == f"model_start:{h.run.calls}"
        raise OSError("Provider failed before returning a response.")

    monkeypatch.setattr(checkpoints.tasks, "reserve_shared", reserve)
    monkeypatch.setattr(checkpoints.tasks, "settle_shared", lambda *args, **kwargs: settlements.append((args, kwargs)))
    monkeypatch.setattr(model_access, "structured", model_double(fail))
    for _ in range(2):
        with pytest.raises(OSError, match="Provider failed"):
            h.ask("treatments:0", contracts.Treatments, "Test only", {"question": "No data", **target_dependency()})
    assert len(reservations) == len(settlements) == 2
    assert [item[1]["status"] for item in settlements] == ["failed", "failed"]
    assert [item[0][2] for item in settlements] == [item[0] for item in reservations]
    assert [item[1]["trace_id"] for item in reservations] == ([str(tree.id) for tree in trees] if traced else [None, None])
    assert [item[1]["trace_url"] for item in reservations] == ([tree.get_url() for tree in trees] if traced else [None, None])
    assert all(item[1]["provider_schema_format"] == "response_json_schema" for item in reservations)
    assert reservations[0][1]["request"] == reservations[1][1]["request"]
    assert all("trace_id" not in item[1]["request"] and "trace_url" not in item[1]["request"] for item in reservations)
