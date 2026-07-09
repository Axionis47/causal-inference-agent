"""Workers preserve raw proposals while registering explicit conditional claims."""

import json
from copy import deepcopy
from uuid import uuid4

import pytest
from historical_analysis import work
from model_double import model_double
from test_scientific_v8 import decision, study

from backend import model_access
from backend.analysis import references
from backend.analysis.contracts import Roles
from backend.context.contracts import Participant, Relationship, SourceReference


@pytest.fixture(autouse=True)
def isolated():
    yield


def invoke(monkeypatch, kind, *, bad_reference=False):
    context, ids, target, treatments, roles = study()
    previous = {"target": target.model_dump(), "treatments": treatments.model_dump()}
    relation_key = "relationships.association"
    if kind == "role":
        proposal = roles
        claim = proposal.assignments[0].account
    else:
        context.relationships["association"] = Relationship(
            value="The covariate and treatment share a bookkeeping association.",
            status="reviewed", kind="association",
            source=SourceReference(kind="message", id=uuid4(), author_id=uuid4()),
            participants=[Participant(kind="column", id=ids[name]) for name in ("t", "c")],
        )
        proposal = work.RelationshipWork(**decision(), relationships=[{
            "reference": relation_key, "effect_ids": ["primary"], "disposition": "not_applicable",
            "rationale": {"statement": "Conditional on its bookkeeping meaning, the association supplies no causal arrow.",
                          "basis": "supplied", "references": [relation_key]},
        }])
        claim = proposal.relationships[0].rationale
    claim.basis = "assumption"
    if bad_reference:
        claim.references.append("unoffered-source")
    original = proposal.model_dump(mode="json")
    source = work.packet(context, {ids["c"]}, previous, "Effect of treatment on outcome")
    source.update(_model="gemini-2.5-flash", _prompt="Typed worker fixture", assumption_namespace="task_owned_")
    task = {"id": str(uuid4()), "kind": kind, "column_ids": [ids["c"]], "scope": {"relationship": relation_key},
            "packet": source, "attempts": []}
    saved = []

    def finish(claimed, output, checks, **kwargs):
        receipt = {"id": str(uuid4()), "output": deepcopy(output), "checks": deepcopy(checks), **kwargs}
        saved.append(receipt)
        task["attempts"].append(receipt)
        return receipt

    monkeypatch.setattr(work.tasks, "view", lambda identity: deepcopy(task))
    monkeypatch.setattr(work.tasks, "claim", lambda *args, **kwargs: object())
    monkeypatch.setattr(work.tasks, "finish", finish)
    monkeypatch.setattr(model_access, "structured", model_double(lambda *args, **kwargs: proposal.model_copy(deep=True)))
    work.run(task["id"])
    assert proposal.model_dump(mode="json") == original
    return saved, original


@pytest.mark.parametrize("kind", ["role", "relationship"])
def test_worker_registers_only_explicit_premise_and_preserves_raw_proposal(monkeypatch, kind):
    saved, original = invoke(monkeypatch, kind)
    assert len(saved) == 1 and saved[0]["status"] == "accepted", saved
    receipt = saved[0]
    assert receipt["checks"][0]["rule"] == "evidence_and_assumption_registration"
    assert all(check["passed"] for check in receipt["checks"])
    entry = receipt["output"]["assumptions"][0]
    assert entry["id"].startswith("task_owned_registered_")
    if kind == "role":
        normalized = receipt["output"]["assignments"][0]["account"]
        initial = original["assignments"][0]["account"]
        assert Roles.model_validate(receipt["output"])
    else:
        normalized = receipt["output"]["relationships"][0]["rationale"]
        initial = original["relationships"][0]["rationale"]
    assert normalized["basis"] == initial["basis"] == "assumption"
    assert normalized["statement"] == initial["statement"] == entry["statement"]
    assert normalized["references"] == [*initial["references"], "assumption:" + entry["id"]]
    assert json.loads(receipt["raw_output"]) == original
    assert entry["id"] not in receipt["raw_output"]


@pytest.mark.parametrize("kind", ["role", "relationship"])
def test_worker_does_not_register_unknown_reference_or_bypass_provenance(monkeypatch, kind):
    saved, _ = invoke(monkeypatch, kind, bad_reference=True)
    assert len(saved) == 2 and all(receipt["status"] == "rejected" for receipt in saved)
    assert all(not receipt["output"]["assumptions"] for receipt in saved)
    assert all(any(check["rule"] == "evidence_authority_and_dependencies" and not check["passed"]
                   for check in receipt["checks"]) for receipt in saved)


def test_provider_prerequisites_offer_only_exact_accepted_decisions():
    _, _, choices = references.selections({"decision_refs": ["decision:target", "decision:treatments"]})
    assert choices["prerequisites"] == ["decision:target", "decision:treatments"]
    _, _, empty = references.selections({})
    assert empty["prerequisites"] == []
