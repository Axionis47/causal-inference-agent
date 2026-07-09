"""Audit prose stays inspectable without becoming downstream scientific context."""

from copy import deepcopy

import pytest

from backend.analysis import packets
from backend.analysis.authority_view import inspect, summary


@pytest.fixture(autouse=True)
def isolated():
    yield


def test_model_packets_preserve_admitted_content_without_hypotheses_or_audit_proposals():
    content = {
        "status": "ready", "summary": "Invented semantic interpretation", "changes": "x", "permits": "y",
        "measurement": {"statement": "Exact measured quantity", "basis": "supplied", "references": ["e1"],
                        "admission": {"status": "admitted", "original_proposal": {"statement": "Invented detail"}}},
        "candidate": {"statement": "Prior purchases modify the effect", "basis": "unknown", "references": [],
                      "admission": {"status": "candidate", "original_proposal": {"statement": "Modifier claim"}}},
        "assumptions": [{"id": "inactive", "statement": "Search interactions", "status": "candidate"},
                        {"id": "relied", "statement": "Consistency", "status": "relied_upon", "scope": {"effect_ids": ["primary"]}}],
        "evidence_records": [{"key": "e1", "statement": {"value": "Exact measured quantity", "qualifications": "Applies only at follow-up"}}],
    }
    before = deepcopy(content)
    view = packets.model_view(content)
    assert content == before
    assert view["measurement"]["statement"] == "Exact measured quantity"
    assert view["measurement"]["references"] == ["e1"]
    assert view["candidate"]["basis"] == "unknown"
    assert "modify" not in str(view) and "Search interactions" not in str(view) and "Invented" not in str(view)
    assert view["assumptions"] == [content["assumptions"][1]]
    assert view["evidence_records"] == content["evidence_records"]


def test_inactive_role_coverage_does_not_expand_later_scientific_packets():
    base = {"target": {"outcome": {"column_ids": ["y"]}},
            "treatments": {"treatments": [{"variable": {"column_ids": ["t"]}}]}}
    for width in (12, 60, 200):
        roles = [{"variable": {"column_ids": [f"noise{i}"]}, "roles": ["unknown"],
                  "proposed_roles": ["effect_modifier"], "operational_use": []} for i in range(width - 2)]
        prior = {**base, "roles": {"assignments": roles, "assumptions": []}}
        assert packets.active(prior) == {"t", "y"}
        assert packets.decisions(prior)["roles"]["assignments"] == []
        # A supported operational role must remain even on an otherwise unknown variable.
        roles[0]["operational_use"] = ["grouping"]
        assert "noise0" in packets.active(prior)


def test_inspector_preserves_attempt_lifecycle_without_counting_audit_as_new_claims():
    admission = {"status": "candidate", "authority": "none", "original_proposal": {"admission": {"status": "admitted"}}}
    receipts = [{"id": "first", "checkpoint": "roles", "status": "accepted", "output": {
        "claim": {"statement": "Possible modifier", "admission": admission},
        "assumptions": [{"id": "candidate", "status": "candidate"}],
    }}, {"id": "second", "checkpoint": "causal", "status": "rejected", "output": {
        "assumptions": [{"id": "candidate", "status": "rejected"}],
    }}]
    view = inspect([*receipts, receipts[0]])
    assert len(view["claims"]) == 1 and view["claims"][0]["receipt_id"] == "first"
    assert [item["entry"]["status"] for item in view["ledger"]] == ["candidate", "rejected"]
    assert summary(receipts[0]["output"])["claims"] == {"admitted": 0, "candidate": 1, "rejected": 0}
