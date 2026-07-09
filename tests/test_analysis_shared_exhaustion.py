"""Saved agent allowances bound recovery, including delegated and uncertain calls."""

from copy import deepcopy
from uuid import UUID

import pytest
import test_analysis as analysis
from sqlalchemy.orm import Session
from test_agent_store import PACKET, REQUEST, turn

from backend import model_access
from backend.analysis import jobs
from backend.analysis.agent_store import Store, SubtaskLimit
from backend.db import engine
from backend.models import AnalysisRun

storage = analysis.storage
boundaries = analysis.boundaries


def test_provider_interruptions_cannot_reset_the_study_allowance(owner, boundaries, monkeypatch):
    calls = 0
    def fail(request):
        nonlocal calls
        calls += 1
        raise OSError("Provider unavailable")
    monkeypatch.setattr(model_access, "invoke_tools", fail)
    _, _, chat = analysis.create(owner)
    run, _ = analysis.prepare(owner, chat)
    for index in range(10):
        if index:
            assert owner.post(f"/api/analyses/{run['id']}/retry").status_code == 202
        jobs.process(UUID(run["id"]))
        saved = analysis.read(owner, run["id"])
        assert saved["calls"] == calls == index + 1
        assert saved["retryable"] is (index < 9)
    assert "exhausted" in saved["error"] and not boundaries
    assert owner.post(f"/api/analyses/{run['id']}/retry").status_code == 409
    jobs.process(UUID(run["id"]))
    assert calls == 10


@pytest.mark.parametrize("status", ["failed", "uncertain"])
def test_view_derives_exhaustion_without_rewriting_saved_history(owner, boundaries, status):
    _, _, chat = analysis.create(owner)
    run, _ = analysis.prepare(owner, chat)
    identity = UUID(run["id"])
    with Session(engine()) as db:
        store = Store(db, identity)
        task = store.open("study", PACKET)
        for _ in range(10):
            attempt = store.reserve(task, REQUEST)
            if status == "failed":
                store.settle(attempt, None, OSError("Interrupted"))
            else:
                store.recover()
        row = db.get(AnalysisRun, identity)
        row.stage, row.status, row.error = "study", "failed", "Interrupted"
        db.commit()
        before = (deepcopy(row.receipts), row.status, row.error, row.calls)
    saved = analysis.read(owner, run["id"])
    assert not saved["retryable"] and "exhausted" in saved["error"]
    assert owner.post(f"/api/analyses/{run['id']}/retry").status_code == 409
    with Session(engine()) as db:
        row = db.get(AnalysisRun, identity)
        assert (row.receipts, row.status, row.error, row.calls) == before


def test_delegated_local_exhaustion_leaves_parent_allowance_available(owner, boundaries):
    _, _, chat = analysis.create(owner)
    run, _ = analysis.prepare(owner, chat)
    with Session(engine()) as db:
        store = Store(db, UUID(run["id"]))
        parent = store.open("study", PACKET)
        child = store.open("study", PACKET, parent=parent, key="optional-evidence", limit=2)
        for _ in range(2):
            attempt = store.reserve(child, REQUEST)
            store.settle(attempt, None, OSError("Interrupted"))
        with pytest.raises(SubtaskLimit):
            store.reserve(child, REQUEST)
        assert store.locked_run().status == "queued" and store.counts("study") == 2
        attempt = store.reserve(parent, REQUEST)
        store.settle(attempt, turn())
        assert store.counts("study") == 3


def test_last_saved_provider_response_can_finish_without_new_allowance(owner, boundaries):
    _, _, chat = analysis.create(owner)
    run, _ = analysis.prepare(owner, chat)
    with Session(engine()) as db:
        store = Store(db, UUID(run["id"]))
        task = store.open("study", PACKET)
        for _ in range(9):
            attempt = store.reserve(task, REQUEST)
            store.settle(attempt, None, OSError("Interrupted"))
        attempt = store.reserve(task, REQUEST)
        store.settle(attempt, turn("publish"))
        row = store.locked_run()
        row.stage, row.status = "study", "failed"
        db.commit()
        assert store.exhaustion(row) is None
        store.tool(task, attempt, 0, "publish", {}, lambda key: store.publish(task, {"study": "Saved work"}, "Accepted", []))
        assert store.exhaustion(row) is None and row.calls == 10
