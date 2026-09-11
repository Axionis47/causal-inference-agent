"""Provider lowering stays weaker than, and separate from, all local task contracts."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from google.genai import _transformers
from pydantic import BaseModel, ValidationError

from causal.design.compile import load_task_table
from causal.design.contracts import DesignIntentV1
from causal.design.semantics import (
    CausalContextV1,
    ColumnSemanticCardV1,
    RoleEvidenceV1,
    RoleLedgerV1,
    SlotAssertionV1,
)
from causal.design.v2 import AgentDesignProposalV2
from causal.shared.agenttask import result_schema
from causal.shared.contracts import ReferenceKind
from causal.shared.envelope import ContextRequirementV1
from causal.shared.gateway import _provider_schema
from causal.shared.validation import validate_references

ROOT = Path(__file__).parents[2]
TASK_SPECS = load_task_table(ROOT / "registries" / "design-tasks.v1.json")
TASKS: tuple[tuple[str, type[BaseModel], bool], ...] = (
    ("intent", DesignIntentV1, False),
    ("semantic_batch", ColumnSemanticCardV1, True),
    ("role_evidence", RoleEvidenceV1, False),
    ("causal_context", CausalContextV1, False),
    ("role_ledger", RoleLedgerV1, False),
    ("method_design", AgentDesignProposalV2, False),
)
CATALOGS = {
    "evidence_ids": ("ev:a", "ev:b"),
    "diagnostic_ids": ("diag:a", "diag:b"),
    "diagnostic_result_ids": ("dr:a", "dr:b"),
    "column_ids": ("col_a", "col_b"),
    "concept_ids": ("concept:a", "concept:b"),
    "graph_edge_ids": ("edge:a", "edge:b"),
    "alternative_ids": ("alt:a", "alt:b"),
    "method_ids": ("aipw", "did"),
    "artifact_ids": ("artifact:a", "artifact:b"),
}
# Fields, enum nodes, enum values, minItems nodes, maxItems nodes, nullable nodes after the
# pinned SDK has expanded definitions. These counts make an accidental complexity regression
# visible without sending a request to the provider.
SDK_COUNTS = {
    "intent": (61, 20, 64, 0, 3, 0),
    "semantic_batch": (80, 23, 87, 0, 3, 13),
    "role_evidence": (46, 21, 84, 1, 3, 0),
    "causal_context": (53, 22, 59, 1, 3, 0),
    "role_ledger": (39, 18, 72, 1, 3, 0),
    "method_design": (43, 17, 67, 1, 4, 1),
}
LEXICAL = frozenset({"minLength", "maxLength", "pattern"})
BOUNDS = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems",
})


def _walk(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from _walk(child)


def _schema(kind: str, model: type[BaseModel], many: bool) -> dict[str, object]:
    requirement_ids = (() if kind not in TASK_SPECS
                       else TASK_SPECS[kind].allowed_requirement_ids)
    return result_schema(
        model, many=many, requirement_ids=requirement_ids, **CATALOGS)


def _sdk_schema(schema: dict[str, object]) -> dict[str, Any]:
    """Run the pinned SDK's real local schema conversion; no client or request is created."""
    converted = _transformers.t_schema(None, deepcopy(schema))
    assert converted is not None
    return converted.model_dump(
        mode="json", by_alias=True, exclude_none=True, exclude_unset=True)


def _keys(schema: Mapping[str, Any], wanted: frozenset[str]) -> list[tuple[str, Any]]:
    return [(key, value) for node in _walk(schema) if isinstance(node, Mapping)
            for key, value in node.items() if key in wanted]


def _counts(schema: Mapping[str, Any]) -> tuple[int, int, int, int, int, int]:
    nodes = [node for node in _walk(schema) if isinstance(node, Mapping)]
    enums = [node["enum"] for node in nodes if isinstance(node.get("enum"), list)]
    return (
        sum(len(node.get("properties", {})) for node in nodes),
        len(enums),
        sum(len(values) for values in enums),
        sum("minItems" in node for node in nodes),
        sum("maxItems" in node for node in nodes),
        sum(node.get("nullable") is True for node in nodes),
    )


@pytest.mark.parametrize(("kind", "model", "many"), TASKS)
def test_all_production_response_schemas_are_lowered_only_for_the_decoder(
    kind: str, model: type[BaseModel], many: bool,
) -> None:
    local = _schema(kind, model, many)
    before = deepcopy(local)

    provider = _provider_schema(local)
    sdk = _sdk_schema(provider)

    assert local == before
    assert _keys(local, LEXICAL)
    assert not _keys(provider, LEXICAL)
    assert not _keys(sdk, LEXICAL)
    assert _keys(provider, BOUNDS) == _keys(local, BOUNDS)
    assert {"complete", "needs_context", "conflict", "refused"} <= {
        value for node in _walk(sdk) if isinstance(node, Mapping)
        for value in node.get("enum", ())
    }
    requirement_ids = TASK_SPECS[kind].allowed_requirement_ids if kind in TASK_SPECS else ()
    enum_values = {value for node in _walk(sdk) if isinstance(node, Mapping)
                   for value in node.get("enum", ())}
    assert not (set(CATALOGS["evidence_ids"]) & enum_values)
    assert set(requirement_ids) <= enum_values
    assert _counts(sdk) == SDK_COUNTS[kind]


def test_long_live_evidence_catalog_does_not_multiply_decoder_enum_states() -> None:
    # These identifiers triggered Vertex's serving-state limit in the live cable-system
    # experiment. The semantic card expands its evidence slot twelve times in the SDK.
    evidence = (
        "ev:kaggle/dataset/description", "ev:kaggle/dataset/title",
        "ev:kaggle/file/rock_the_vote.csv/description",
        "ev:kaggle/column/rock_the_vote.csv/cable_system_id/description",
        "ev:kaggle/column/rock_the_vote.csv/randomization_stratum/description",
        "tableprofile:an-82bd653613239d1c:0548be1486238151#/columns/cable_system_id",
        "tableprofile:an-82bd653613239d1c:0548be1486238151#/columns/randomization_stratum",
    )
    local = result_schema(ColumnSemanticCardV1, many=True, evidence_ids=evidence)
    before = deepcopy(local)
    sdk = _sdk_schema(_provider_schema(local))
    previous = deepcopy(local)
    for node in _walk(previous):
        if isinstance(node, dict):
            node.pop("x-causal-reference-role", None)
    strict_sdk = _sdk_schema(_provider_schema(previous))
    assert local == before
    assert sum(node.get("enum") == list(evidence) for node in _walk(strict_sdk)
               if isinstance(node, Mapping)) == 13
    assert not any(node.get("enum") == list(evidence) for node in _walk(sdk)
                   if isinstance(node, Mapping))
    assert _counts(sdk)[2] < _counts(strict_sdk)[2] - len(evidence) * 12
    assert {"complete", "needs_context", "conflict", "refused", "evidenced", "hypothesis"} <= {
        value for node in _walk(sdk) if isinstance(node, Mapping)
        for value in node.get("enum", ())}
    invalid = {"value": "cable system", "status": "evidenced",
               "evidence_ids": ["ev:invented"]}
    issues = validate_references(SlotAssertionV1, invalid, {ReferenceKind.EVIDENCE: evidence})
    assert [(issue.code, issue.json_path) for issue in issues] == [
        ("unresolved_evidence", "/evidence_ids/0")]


def test_numeric_and_array_constraints_are_not_part_of_lexical_lowering() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "items": {"type": "array", "minItems": 1, "maxItems": 4,
                      "items": {"type": "string", "minLength": 1, "maxLength": 200}},
        },
    }
    lowered = _provider_schema(schema)
    sdk = _sdk_schema(lowered)

    assert _keys(lowered, BOUNDS) == _keys(schema, BOUNDS)
    assert sorted(_keys(sdk, BOUNDS)) == sorted(_keys(schema, BOUNDS))
    assert not _keys(lowered, LEXICAL)


@pytest.mark.parametrize("kind", tuple(ReferenceKind))
def test_only_long_evidence_and_artifact_catalogs_are_omitted(kind: ReferenceKind) -> None:
    schema: dict[str, object] = {
        "type": "array", "minItems": 1, "maxItems": 2,
        "x-causal-reference-kind": kind.value, "x-causal-reference-role": "reference",
        "items": {"type": "string", "enum": ["allowed:a", "allowed:b"]},
    }
    provider = _provider_schema(schema)
    expected_items: dict[str, object] = {"type": "string"}
    if kind not in {ReferenceKind.EVIDENCE, ReferenceKind.ARTIFACT}:
        expected_items["enum"] = ["allowed:a", "allowed:b"]
    assert provider == {"type": "array", "minItems": 1, "maxItems": 2,
                        "items": expected_items}


def _requirement(requirement_id: str, scope_id: str) -> dict[str, object]:
    return {
        "requirement_id": requirement_id,
        "registry_version": "context-requirements.v1",
        "scope_kind": "design",
        "scope_id": scope_id,
        "fact_required": "An admitted assignment mechanism.",
        "why_required": "Method eligibility depends on it.",
        "decisions_blocked": ["method_selection"],
        "criticality": "blocking",
        "acceptable_evidence_types": ["source_statement"],
        "required_support": "direct",
        "methods_required_for": ["sharp_rdd"],
        "attempted_evidence": [],
        "user_may_know": True,
        "expected_answer_schema": "choice:randomized|policy_cutoff",
        "missing_action": "ask_user",
    }


def test_full_local_contract_still_rejects_relaxed_length_and_reference_values() -> None:
    with pytest.raises(ValidationError, match="string_too_long"):
        ContextRequirementV1.model_validate_json(json.dumps(
            _requirement("design.assignment_mechanism", "x" * 201)))

    invalid = _requirement("design.not_registered", "design")
    issues = validate_references(
        ContextRequirementV1, invalid,
        {ReferenceKind.REQUIREMENT: ("design.assignment_mechanism",)},
    )
    assert [(issue.code, issue.json_path, issue.artifact_ids) for issue in issues] == [
        ("unresolved_requirement", "/requirement_id", ("design.not_registered",))]
