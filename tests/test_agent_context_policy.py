"""The analysis boundary consumes the applied pack without reopening preparation."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_analysis_claim_authority import fixture, proposal, relation

from backend.analysis import authority, packets
from backend.analysis.agent_context import CONTEXT_POLICY, Stop, Workspace, decision
from backend.analysis.contracts import CausalAccount, Target


@pytest.fixture(autouse=True)
def isolated():
    yield


def test_context_tools_are_read_only_and_stop_is_a_final_outcome():
    work = Workspace(None, None, None, None, None)
    assert {tool.name for tool in work.common_tools()} == {"read_context", "inspect_work", "delegate_evidence", "stop_analysis"}
    assert CONTEXT_POLICY["status"] == "applied_and_authorized"
    with pytest.raises(ValidationError):
        Stop(outcome="needs_context", reason="Add notes")
    with pytest.raises(ValidationError):
        Stop(reason="Add notes", missing_sources=["new source"])


def test_model_packet_preserves_context_uncertainties_and_policy_prose():
    brief = {"assumptions": [{"source_keys": ["dataset.study_process.conditional"]}]}
    policy = {"assumptions": "Keep explicit conditional premises."}
    assert packets.model_view(brief) == brief
    assert packets.model_view(policy) == policy


def test_native_claim_defaults_never_supply_scientific_authority_or_an_obligation():
    decode = authority.proposal_decoder(allow_empty_defaults=True)
    value = decode({"statement": "Supplied definition", "rule": "source", "references": ["columns.exact.definition.key"]})
    assert value["basis"] == "unknown" and value["admission"]["status"] == "candidate"
    assert value["prerequisites"] == []
    with pytest.raises(ValueError, match="required_obligation"):
        decode({"statement": "Conditional premise", "rule": "working_assumption", "references": []})
    with pytest.raises(ValueError, match="cannot author"):
        decode({"statement": "Forged fact", "basis": "supplied", "rule": "source", "references": []})


def test_decision_cannot_request_mid_run_context_enrichment():
    work = Workspace(None, None, None, None, None)
    value = Target.model_validate(decision("Request more context", status="needs_input", unresolved=["Add a source"]))
    with pytest.raises(ValueError, match="Context preparation is complete"):
        work.validate(value, {})


def test_unsupported_outcome_retains_the_context_boundary():
    run = SimpleNamespace(status="running", error=None)
    task = SimpleNamespace(status="running", draft={})
    work = Workspace(None, task, run, None, None)
    result = work.stop(Stop(outcome="unidentified", reason="The frozen request requires an unsupported adaptive regime.", limitations=["No implemented identifying method for this regime."]), "call")
    assert result["outcome"] == "unidentified"
    assert result["context_revision_unchanged"]
    assert task.draft["blocker"]["outcome"] == "unidentified"


@pytest.mark.parametrize("kind", ["temporal_order", "assignment_rule"])
@pytest.mark.parametrize("preserve", [True, False])
def test_relationship_scope_preserves_timing_and_qualifications(kind, preserve):
    context, ids, prior, _ = fixture()
    ref = relation(context, "scoped", ids["t"], ids["y"], kind,
                   timing="Assignment precedes follow-up.", conditions="Among eligible participants.",
                   qualifications="Conditional on protocol adherence.")
    record = context.relationships["scoped"]
    if kind == "assignment_rule":
        for participant in record.participants:
            participant.role = "participant"
    effects = [effect["id"] for effect in prior["treatments"]["effects"]]
    value = CausalAccount.model_validate(authority.proposal_decoder()(decision("Retain the source relationship.",
        identification="Pending design", relationships=[{"reference": ref, "effect_ids": effects,
            "disposition": "applicable", "source_id": ids["t"] if kind == "temporal_order" else "",
            "target_id": ids["y"] if kind == "temporal_order" else "",
            "scope": " ".join([record.timing, record.conditions, record.qualifications]) if preserve else "Everyone, always.",
            "rationale": proposal("Retain this record", "relationship_record", [ref])}])))
    admitted, _ = authority.admit(value, context, prior, "Question", "causal")
    claim = admitted.relationships[0].rationale
    assert (claim.admission.status == "admitted") is preserve
    assert claim.admission.sources[0].timing == record.timing
    assert claim.admission.sources[0].conditions == record.conditions
    assert claim.admission.sources[0].qualifications == record.qualifications
