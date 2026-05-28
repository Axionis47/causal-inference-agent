"""Analysis-owned proposal encoding and decoding policy. No provider calls or budgets."""

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from backend import model_access
from backend.analysis import provider_fields
from backend.analysis.authority import RULES


def thinking_config(model: str, output_limit: int) -> dict[str, Any] | None:
    """Explicit analysis allocations; provider thinking budgets are soft limits."""
    budget = {4096: 1024, 8192: 2048, 16384: 2048}.get(output_limit)
    if not model_access.supports_thinking(model) or budget is None:
        return None
    return {"thinking_budget": budget, "include_thoughts": False}


def provider_schema_format(choices: dict[str, list[str]] | None = None) -> str:
    return "response_json_schema" if choices and choices.get("_json_schema") else "response_schema"


def provider_schema(schema: type[BaseModel], choices: dict[str, list[str]] | None = None) -> dict[str, Any]:
    native = schema.model_json_schema()
    authority_policy = bool(choices and choices.get("_authority_policy"))
    if authority_policy:
        for name, entry in native.get("$defs", {}).items():
            props = entry.get("properties", {})
            if name in {"Claim", "CausalEdge"}:
                props.pop("basis", None)
                props.pop("admission", None)
                props["rule"] = {"type": "string", "enum": list(RULES)}
                props["required_obligation"] = {"type": "string", "description": "Empty unless proposing a necessary working assumption for an exact causal identification obligation; never a source-authority label."}
                if choices is not None and "_claim_refs" in choices:
                    props["references"] = {"type": "array", "items": {"type": "string", "enum": choices["_claim_refs"]}} if choices["_claim_refs"] else {"type": "array", "maxItems": 0, "items": {"type": "string"}}
                entry["required"] = list(props)
                if choices and choices.get("_proposal_defaults"):
                    entry["required"] = [key for key in entry["required"] if key not in {"prerequisites", "required_obligation"}]
            if name == "RoleAssignment":
                for field in ("proposed_roles", "coverage_status", "outcome_proxy", "proxy_evidence"):
                    props.pop(field, None)
            if name == "Assumption":
                for field in ("status", "scope", "original_proposal", "admission_reason", "required_obligation"):
                    props.pop(field, None)
        # Public audit contracts are not provider proposal alternatives.
        definitions_to_visit: list[str] = []
        def refs(value: Any) -> None:
            if isinstance(value, dict):
                if "$ref" in value:
                    definitions_to_visit.append(value["$ref"].split("/")[-1])
                for key, item in value.items():
                    if key != "$defs":
                        refs(item)
            elif isinstance(value, list):
                for item in value:
                    refs(item)
        refs(native)
        retained: set[str] = set()
        while definitions_to_visit:
            name = definitions_to_visit.pop()
            if name not in retained:
                retained.add(name)
                refs(native["$defs"][name])
        if "$defs" in native:
            native["$defs"] = {name: entry for name, entry in native["$defs"].items() if name in retained}
    def definition(name: str, converted: dict[str, Any]) -> dict[str, Any]:
        if name == "EffectRequest" and choices is not None and "_effect_refs" in choices:
            if not choices["_effect_refs"]:
                raise ValueError("An effect-reference schema requires a nonempty accepted target registry.")
            for field in ("contrast", "priority"):
                converted["properties"].pop(field, None)
            converted["required"] = [field for field in converted.get("required", []) if field not in {"contrast", "priority"}]
            converted["properties"]["id"].update(
                enum=choices["_effect_refs"],
                description="Select the exact accepted requested-effect identity. The harness restores its question and priority without changing their source authority.",
            )
        if name == "Assumption" and choices and choices.get("_assumption_namespace"):
            converted["properties"]["id"]["description"] = "NEW assumptions only. Use this exact namespace prefix: " + choices["_assumption_namespace"][0] + ". Existing assumptions are references, never re-authored here."
            converted["properties"]["id"]["pattern"] = "^" + re.escape(choices["_assumption_namespace"][0]) + "[A-Za-z0-9_]+$"
        if name == "RoleAssignment":
            converted["properties"].pop("assignment_event_key", None)
            if choices is not None and "_variable_refs" in choices:
                if not choices["_variable_refs"]:
                    raise ValueError("A reference-only role schema requires a nonempty accepted variable registry.")
                converted["properties"]["variable"] = {
                    "type": "object", "properties": {"id": {"type": "string", "enum": choices["_variable_refs"]}},
                    "required": ["id"],
                    "description": "Select one exact established variable identity only. The harness restores its accepted binding; do not repeat or alter its other fields.",
                }
        if name == "RoleAssignment" and choices and choices.get("effect_ids"):
            converted["required"] = sorted(set(converted.get("required", [])) | {"effect_ids", "treatment_ids", "time_scope"})
            for field in ("effect_ids", "treatment_ids"):
                converted["properties"][field]["minItems"] = 1
            # Historical fields remain readable; the harness owns classification.
            for field in ("outcome_proxy", "proxy_evidence"):
                converted["properties"].pop(field, None)
        return converted

    encoded = model_access.encode_schema(native, choices=choices,
        schema_format=provider_schema_format(choices),
        explicit_fields=bool(choices and choices.get("_explicit_fields")),
        definition_transform=definition)
    if choices and choices.get("_source_fields"):
        def constrain_source(node: Any, field: str = "") -> None:
            if isinstance(node, list):
                for child in node:
                    constrain_source(child, field)
            elif isinstance(node, dict):
                props = node.get("properties", {})
                if "rule" in props and "references" in props:
                    predicate = {"meaning": "measurement_definition", "question": "requested_effect"}.get(field, field)
                    refs = choices.get("_source:" + predicate, [])
                    props["rule"] = {"type": "string", "enum": ["source", "unknown"] if refs else ["unknown"]}
                    props["references"] = {"type": "array", "maxItems": 1 if refs else 0,
                                           "items": {"type": "string", **({"enum": refs} if refs else {})}}
                    props["references"]["description"] = "For source, select exactly one eligible record. For unknown, leave empty."
                for key, child in node.items():
                    if key == "properties":
                        for name, entry in child.items():
                            constrain_source(entry, name)
                    else:
                        constrain_source(child, field)
        constrain_source(encoded)
    return provider_fields.compact(encoded) if choices and choices.get("_compact_fields") else encoded


def prepare[T: BaseModel](
    schema: type[T], system: str, payload: dict[str, object], model: str, *,
    images: list[bytes] | None = None, max_output_tokens: int = 32768,
    choices: dict[str, list[str]] | None = None, thinking_budget: int | None = None,
    reference_decoder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> model_access.PreparedModelRequest[T]:
    reference_modes = [name for name in ("_variable_refs", "_effect_refs") if choices and choices.get(name)]
    decoder_required = bool(reference_modes) or bool(choices and choices.get("_authority_policy"))
    def preflight() -> None:
        # Preserve validation timing: the caller reserves before invoking transport.
        if len(reference_modes) > 1 or decoder_required != (reference_decoder is not None):
            raise ValueError("Finite reference output requires its explicit accepted-registry decoder, and vice versa; select exactly one reference mode.")
    decoder = reference_decoder
    if choices and choices.get("_compact_fields"):
        canonical = provider_schema(schema, {key: value for key, value in choices.items() if key != "_compact_fields"})

        def decode(value: dict[str, Any]) -> dict[str, Any]:
            expanded = provider_fields.expand(value, canonical)
            return reference_decoder(expanded) if reference_decoder else expanded

        decoder = decode
    return model_access.prepare(schema, system, payload, model, images=images,
        max_output_tokens=max_output_tokens, thinking_budget=thinking_budget,
        encoded_schema=provider_schema(schema, choices), schema_format=provider_schema_format(choices),
        reference_decoder=decoder, preflight=preflight)
