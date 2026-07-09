"""Owner-scoped shared-attempt visibility without loading large artifacts in polling views."""

from uuid import UUID

import test_analysis as analysis
from conftest import PASSWORD
from historical_analysis import tasks
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.context.service import fingerprint
from backend.db import engine
from backend.models import AnalysisRun, AnalysisTaskAttempt

storage = analysis.storage
boundaries = analysis.boundaries


def test_shared_provider_attempts_expose_charged_states_and_exact_artifacts(owner, boundaries):
    _, _, chat = analysis.create(owner)
    prepared, _ = analysis.prepare(owner, chat)
    run_id = UUID(prepared["id"])
    raw = '{"summary":"' + "x" * 143_000
    request = {"system": "Synthetic proposal", "payload": {"evidence": "source"}, "schema": {"type": "object"}}
    with Session(engine()) as db:
        run = db.get(AnalysisRun, run_id)
        for status in ("failed", "uncertain", "accepted", "reserved"):
            key = f"{status}:0"
            tasks.reserve_shared(db, run, key, fingerprint({"evidence": "source"}),
                                 {"request": request, "trace_id": "saved-trace"}, obligation_key=status)
            if status != "reserved":
                tasks.settle_shared(db, run_id, key, status=status,
                                    receipt={"summary": "Typed proposal, not scientific acceptance"} if status == "accepted" else None,
                                    raw_output=raw if status == "failed" else None,
                                    metrics={"error_type": "OSError"} if status == "failed" else None)
        db.commit()
    saved = analysis.read(owner, str(run_id))
    shared = saved["shared_attempts"]
    assert len(shared) == saved["budgets"]["shared_calls"] == 4
    assert [item["status"] for item in shared] == ["failed", "uncertain", "accepted", "reserved"]
    assert [item["number"] for item in shared] == [1, 2, 3, 4]
    assert all(item["request_hash"] == fingerprint(request) and item["has_request"] for item in shared)
    assert [item["has_proposal"] for item in shared] == [False, False, True, False]
    assert [item["has_raw_output"] for item in shared] == [True, False, False, False]
    assert all(not {"request", "raw_output", "proposal"} & item.keys() for item in shared)
    assert raw not in str(saved)
    assert all(checkpoint["status"] != "accepted" for checkpoint in saved["checkpoints"])
    for summary in shared:
        response = owner.get(f"/api/analyses/{run_id}/provider-attempts/{summary['id']}")
        assert response.status_code == 200, response.text
        detail = response.json()
        assert {key: detail[key] for key in summary} == summary
        assert detail["request"] == request
        assert detail["raw_output"] == (raw if summary["status"] == "failed" else None)
        assert (detail["proposal"] is not None) == (summary["status"] == "accepted")
    assert shared[0]["metrics"]["error_type"] == "OSError"
    assert shared[0]["metrics"]["trace_id"] == "saved-trace"
    assert shared[-1]["completed_at"] is None
    assert analysis.read(owner, str(run_id))["shared_attempts"] == shared


def test_provider_artifacts_require_the_run_owner_and_exclude_foreign_or_worker_attempts(owner, boundaries):
    _, _, chat = analysis.create(owner)
    first, _ = analysis.prepare(owner, chat)
    first_id = UUID(first["id"])
    _, _, second_chat = analysis.create(owner)
    second, _ = analysis.prepare(owner, second_chat)
    with Session(engine()) as db:
        run = db.get(AnalysisRun, first_id)
        tasks.reserve_shared(db, run, "old:0", "a" * 64)
        db.commit()
        shared_id = db.scalar(select(AnalysisTaskAttempt.id).where(AnalysisTaskAttempt.run_id == first_id))
    saved = analysis.read(owner, first["id"])["shared_attempts"][0]
    assert not saved["has_request"] and not saved["has_proposal"] and not saved["has_raw_output"]
    task = tasks.ensure(first_id, key="screen:one", checkpoint="target", kind="screen", column_ids=["one"],
                        scope={}, reason="Synthetic ownership check", dependency_receipt_ids=[], packet={})
    worker = tasks.claim(UUID(task["id"]))
    assert owner.get(f"/api/analyses/{first_id}/provider-attempts/{worker.attempt_id}").status_code == 404
    assert owner.get(f"/api/analyses/{second['id']}/provider-attempts/{shared_id}").status_code == 404
    assert owner.get(f"/api/analyses/{first_id}/provider-attempts/{shared_id}").status_code == 200
    assert owner.post("/api/auth/login", json={"email": "other@example.com", "password": PASSWORD}).status_code == 200
    assert owner.get(f"/api/analyses/{first_id}").status_code == 404
    assert owner.get(f"/api/analyses/{first_id}/provider-attempts/{shared_id}").status_code == 404
