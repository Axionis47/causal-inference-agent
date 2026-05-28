"""Human review returns to computation without rerunning accepted upstream work."""

from uuid import UUID

import test_analysis as analysis

from backend.analysis import jobs

storage = analysis.storage
boundaries = analysis.boundaries


def test_correction_preserves_science_budget_and_requires_new_exact_approval(owner, boundaries):
    saved = analysis.prepared(owner)
    body = {"snapshot_hash": saved["snapshot_hash"], "program_hash": saved["programs"][0]["hash"], "reason": "Clarify the generated program before approval."}
    path = f"/api/analyses/{saved['id']}/revise-program"
    reply = owner.post(path, json=body)
    assert reply.status_code == 202, reply.text
    assert owner.post(path, json=body).status_code == 202
    assert len(boundaries) == 1
    jobs.process(UUID(saved["id"]))
    revised = analysis.read(owner, saved["id"])
    assert revised["status"] == "awaiting_approval", revised["error"]
    assert revised["snapshot_hash"] == saved["snapshot_hash"] and revised["snapshot"] == saved["snapshot"]
    assert [a["handoff_hash"] for a in revised["agents"][:2]] == [a["handoff_hash"] for a in saved["agents"][:2]]
    assert [a["model_calls"] for a in revised["agents"][:2]] == [a["model_calls"] for a in saved["agents"][:2]]
    assert revised["calls"] == saved["calls"] + 3
    assert len(revised["programs"]) == 2 and revised["programs"][0]["execution"] is None
    assert revised["programs"][0]["rejected_reason"] == body["reason"]
    assert revised["programs"][1]["diff"] and len(boundaries) == 1
    assert owner.post(f"/api/analyses/{saved['id']}/approve", json={key: body[key] for key in ("snapshot_hash", "program_hash")}).status_code == 409
    complete = analysis.approval(owner, saved["id"])
    assert complete["status"] == "succeeded", complete["error"]
    assert len(boundaries) == 2
    assert owner.post(path, json=body).status_code == 409


def test_correction_queue_failure_is_atomic(owner, boundaries, monkeypatch):
    from backend.analysis import delivery
    saved = analysis.prepared(owner)
    monkeypatch.setattr(delivery, "enqueue", lambda *args: (_ for _ in ()).throw(OSError("Queue unavailable")))
    response = owner.post(f"/api/analyses/{saved['id']}/revise-program", json={"snapshot_hash": saved["snapshot_hash"], "program_hash": saved["programs"][0]["hash"], "reason": "Correct code"})
    assert response.status_code == 503
    assert analysis.read(owner, saved["id"]) == saved
