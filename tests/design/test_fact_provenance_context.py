"""The actual revision-two accepted facts remain usable without becoming fabricated evidence."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from causal.design import compile as compiler
from causal.design import contracts, packs, resolution_v2, v2
from causal.design.askgate import AcceptedFactV1
from causal.design.harness_base import HarnessBase
from causal.shared.validation import parse_strict

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/shared/fixtures/rct_revision2_fact_provenance.json"


def captured():
    case = json.loads(FIXTURE.read_text())
    facts = [parse_strict(AcceptedFactV1, row) for row in case["facts"]]
    return case, facts


def harness_for(monkeypatch, case, facts, evidence):
    harness = object.__new__(HarnessBase)
    harness.deps = SimpleNamespace(conn=SimpleNamespace(execute=lambda *_: SimpleNamespace(fetchall=list)))
    monkeypatch.setattr(harness, "_accepted_facts", lambda _: {(row.requirement_id, row.scope_id): row for row in facts})
    monkeypatch.setattr(harness, "_evidence", lambda _: evidence)
    state = {"analysis_id": case["analysis_id"], "design_revision": 2}
    return harness, state


def test_exact_revision_two_provenance_is_retained_without_new_evidence_authority(monkeypatch):
    case, facts = captured()
    evidence = deepcopy(case["offered_evidence"])
    harness, state = harness_for(monkeypatch, case, facts, evidence)
    before = [row.model_dump(mode="json") for row in facts]
    packet = harness._prerequisite_context(state)
    assert packet["settled_requirements"] == []  # No fabricated inherited requirement rows.
    for row, fact in zip(packet["accepted_facts"], facts, strict=True):
        assert row["value"] == fact.value and row["accepted_fact_id"] == fact.accepted_fact_id
        assert row["provenance"] == {
            "source_kind": fact.source_kind, "origin_revision": fact.origin_revision,
            "origin_reference_id": fact.origin_reference_id,
            "origin_reference_hash": fact.origin_reference_hash,
            "evidence_ids": list(fact.evidence_ids)}
    answers = {key for fact in facts if fact.source_kind == "user" for key in fact.evidence_ids
               if key.startswith("ua:") and key in evidence}
    assert set(packet["accepted_answer_evidence_ids"]) == answers and answers
    assert not any(row.accepted_fact_id in evidence for row in facts)
    assert "not citable evidence" in packet["contract"]
    assert "source_evidence_ids" in packet["contract"]
    assert evidence == case["offered_evidence"]
    assert [row.model_dump(mode="json") for row in facts] == before


@pytest.mark.parametrize("availability", ("absent", "empty"))
def test_provenance_survives_when_answer_is_not_available_for_citation(monkeypatch, availability):
    case, facts = captured()
    evidence = deepcopy(case["offered_evidence"])
    answers = {key for fact in facts if fact.source_kind == "user" for key in fact.evidence_ids}
    for key in answers:
        if availability == "empty":
            evidence[key] = ""
        else:
            evidence.pop(key, None)
    harness, state = harness_for(monkeypatch, case, facts, evidence)
    packet = harness._prerequisite_context(state)
    assert packet["accepted_answer_evidence_ids"] == []
    user = next(row for row in packet["accepted_facts"] if row["provenance"]["source_kind"] == "user")
    assert set(user["provenance"]["evidence_ids"]) == answers


def test_accepted_exact_grain_compiles_without_redundant_intent_source_interpretation():
    case, facts = captured()
    draft = deepcopy(case["intent_payload"])
    draft["source_interpretations"] = []  # Test the now-explicit accepted-fact reuse contract.
    intent = parse_strict(contracts.DesignIntentV1, draft)
    proposal = parse_strict(v2.AgentDesignProposalV2, case["proposal_payload"])
    rows = resolution_v2.compile_proposal_facts(proposal, evidence=case["offered_evidence"],
        requirement_templates=packs.load_requirement_templates(ROOT / "registries/context-requirements.v1.json"),
        accepted_facts=facts, candidate_grain=intent.candidate_grain,
        source_interpretations=intent.source_interpretations)
    grain = next(row for row in rows if row.fact_id == "grain")
    accepted = next(row for row in facts if row.requirement_id == "design.table_grain")
    assert grain.executable and grain.value == accepted.value == "one_row_per_unit"
    assert grain.source_artifact_ids == (accepted.origin_reference_id,)
    assert grain.supporting_evidence_ids == accepted.evidence_ids
    spec = compiler.load_task_table(ROOT / "registries/design-tasks.v1.json")["intent"]
    prompt = compiler.render_prompt(spec, ROOT, {})
    assert "omit the redundant grain source_interpretation" in prompt
    assert "accepted_fact_id" in prompt and "not citable evidence" in prompt
