"""Human rejection stops an exact unapproved packet without revising science."""

from uuid import UUID

import pytest
import test_analysis as analysis
from sqlalchemy.orm import Session

from backend.analysis import artifacts, jobs, service
from backend.db import engine
from backend.models import AnalysisRun

storage = analysis.storage
boundaries = analysis.boundaries


def body(saved):
    return {"snapshot_hash": saved["snapshot_hash"], "program_hash": saved["programs"][-1]["hash"],
            "reason": "The proposed program substitutes an interval for undefined inference."}


def test_rejection_is_bound_saved_idempotent_and_available_to_future_chat(owner, boundaries):
    saved = analysis.prepared(owner)
    with Session(engine()) as db:
        before = artifacts.binding(db.get(AnalysisRun, UUID(saved["id"])))
    request = body(saved)
    path = f"/api/analyses/{saved['id']}/reject"
    response = owner.post(path, json=request)
    assert response.status_code == 200, response.text
    rejected = response.json()
    assert rejected["status"] == "blocked" and not rejected["retryable"]
    assert rejected["snapshot"] == saved["snapshot"]
    assert rejected["programs"][0] == {**saved["programs"][0], "rejected_reason": request["reason"]}
    assert rejected["calls"] == saved["calls"] and len(boundaries) == 1
    receipt = rejected["events"][-1]
    assert receipt["kind"] == "program_rejected"
    assert receipt["detail"]["reason"] == request["reason"]
    assert owner.post(path, json=request).json() == rejected
    assert owner.post(path, json={**request, "reason": "Different review"}).status_code == 409
    assert owner.post(f"/api/analyses/{saved['id']}/approve", json={k: request[k] for k in ("snapshot_hash", "program_hash")}).status_code == 409
    jobs.process(UUID(saved["id"]))
    assert analysis.read(owner, saved["id"]) == rejected and len(boundaries) == 1
    name = "review-rejection-0.json"
    content = owner.get(f"/api/analyses/{saved['id']}/files/{name}")
    assert content.status_code == 200 and content.json()["reason"] == request["reason"]
    with Session(engine()) as db:
        after = artifacts.binding(db.get(AnalysisRun, UUID(saved["id"])))
    assert name not in before["files"] and name in after["files"]
    assert [e["sequence"] for e in rejected["events"]] == list(range(1, len(rejected["events"]) + 1))


def test_rejection_rejects_stale_hashes_blank_reason_and_already_approved_execution(owner, boundaries):
    saved = analysis.prepared(owner)
    request = body(saved)
    path = f"/api/analyses/{saved['id']}/reject"
    for field in ("snapshot_hash", "program_hash"):
        assert owner.post(path, json={**request, field: "0" * 64}).status_code == 409
    assert owner.post(path, json={**request, "reason": "   "}).status_code == 422
    assert analysis.read(owner, saved["id"]) == saved
    result = analysis.approval(owner, saved["id"])
    assert owner.post(path, json=request).status_code == 409
    assert analysis.read(owner, saved["id"]) == result


def test_rejection_receipt_and_event_roll_back_together(owner, boundaries, monkeypatch):
    saved = analysis.prepared(owner)
    original = service.emit

    def broken(db, run, key, stage, kind, summary, detail=None):
        original(db, run, key, stage, kind, summary, detail)
        if kind == "program_rejected":
            raise RuntimeError("Injected publication failure")

    monkeypatch.setattr(service, "emit", broken)
    with pytest.raises(RuntimeError, match="Injected publication failure"):
        owner.post(f"/api/analyses/{saved['id']}/reject", json=body(saved))
    assert analysis.read(owner, saved["id"]) == saved
