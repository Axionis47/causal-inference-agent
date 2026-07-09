"""Compact preserved failures test bookkeeping only, not semantic endorsement."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from google.genai import types
from test_model_observability import client
from test_scientific_v8 import decision

from backend import model_access
from backend.analysis import domain, provider, references, registration
from backend.analysis.contracts import CausalAccount, CausalEdge, Claim, Decision
from backend.context.contracts import ContextContent, empty_content

FIXTURES = Path(__file__).parent / "fixtures" / "analysis_registration_v13"
COMPONENTS = json.loads((FIXTURES / "components.json").read_text())


@pytest.fixture(autouse=True)
def isolated():
    yield


class EdgeDecision(Decision):
    edge: CausalEdge


class ClaimDecision(Decision):
    claim: Claim


def provenance(value, context, prior):
    return next(gate for gate in domain.checks(value, context, prior) if gate.rule == "evidence_authority_and_dependencies")


def test_actual_opaque_edge_registers_identically_after_rename_reorder_and_scope_expansion():
    original = COMPONENTS["opaque"]
    context = ContextContent.model_validate(original["context"])
    proposal = EdgeDecision(**decision(), edge=original["edge"])
    assert not provenance(proposal, context, {}).passed
    first, notes = registration.normalize(proposal, context, {}, "causal_")
    expanded = context.model_copy(deep=True)
    for position, column in enumerate(expanded.columns.values()):
        column.identity.name = f"Renamed display label {position}"
    expanded.columns.update(empty_content(uuid4(), ["unit", "event_time"]).columns)
    expanded.columns = dict(reversed(list(expanded.columns.items())))
    second, _ = registration.normalize(proposal, expanded, {}, "causal_")
    assert first == second and len(first.assumptions) == 1 and notes
    assert first.edge.source == proposal.edge.source and first.edge.target == proposal.edge.target
    assert first.edge.basis == proposal.edge.basis == "assumption"
    assert first.assumptions[0].statement == first.edge.explanation == proposal.edge.explanation
    assert first.edge.references == [*proposal.edge.references, "assumption:" + first.assumptions[0].id]
    assert first.assumptions[0].depends_on == sorted(proposal.edge.references)
    assert provenance(first, context, {}).passed and proposal.model_dump()["assumptions"] == []


def test_actual_wide_role_keeps_exact_claim_and_separates_accepted_dependency():
    proposal = ClaimDecision(**decision(), claim=COMPONENTS["wide"]["account"])
    context = ContextContent()
    prior = {"treatments": {"assumptions": []}}
    assert not provenance(proposal, context, prior).passed
    normalized, _ = registration.normalize(proposal, context, prior, "role_")
    assert normalized.claim.statement == proposal.claim.statement
    assert normalized.claim.basis == "supplied" and normalized.claim.references == ["question"]
    assert normalized.claim.prerequisites == ["decision:treatments"]
    assert not normalized.assumptions and provenance(normalized, context, prior).passed
    assert not provenance(normalized, context, {}).passed, "An unoffered decision remains invalid after normalization."


def test_actual_factorial_replay_separates_reference_types_without_certifying_causal_entailment():
    original = COMPONENTS["factorial"]
    context = ContextContent.model_validate(original["context"])
    prior = {"treatments": {"assumptions": []}}
    proposal = EdgeDecision(**decision(), edge=original["edge"])
    normalized, _ = registration.normalize(proposal, context, prior, "causal_")
    assert provenance(normalized, context, prior).passed
    assert normalized.edge.explanation == proposal.edge.explanation
    assert "as implied by the request to estimate its effect" in normalized.edge.explanation
    assert normalized.edge.basis == proposal.edge.basis == "supplied"
    assert normalized.edge.prerequisites == ["decision:treatments"]
    assert not normalized.assumptions, "Registration cannot silently relabel or repair a supplied causal assertion."


def test_actual_truncated_opaque_response_is_preserved_and_not_repaired_or_replayed_wholesale(monkeypatch):
    raw = (FIXTURES / "opaque_truncated.txt").read_text()
    response = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(parts=[types.Part(text=raw)]))])
    calls = client(monkeypatch, response)
    with pytest.raises(model_access.InvalidModelOutput) as caught:
        model_access.structured(provider.prepare(CausalAccount, "Parse the saved malformed fixture", {}, "fixture-model"))
    persisted, metrics = model_access.response_artifacts(caught.value)
    assert persisted == raw and len(calls) == 1
    assert metrics["finish_reason"] is None and metrics["usage"] is None, "v13 did not establish why the response ended."
    correction = references.correction({"invalid_submission": raw},
        [{"rule": "typed_submission", "passed": False, "detail": str(caught.value)}], "historical:opaque:causal:0")
    assert raw not in str(correction) and "rejected_fields" not in correction
    assert len(json.dumps(correction)) < 4000
