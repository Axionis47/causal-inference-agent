"""Interrupted native turns resume saved work and retain spent allowances."""

from uuid import UUID, uuid4

import pytest
import test_analysis as analysis
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_agents_journey import agent_detail, native_script, selected_call, tool_turn

from backend import model_access
from backend.analysis import jobs, prompts
from backend.db import engine
from backend.models import AnalysisTask, AnalysisTaskAttempt

storage = analysis.storage
boundaries = analysis.boundaries


@pytest.mark.parametrize("interruption", [OSError, KeyboardInterrupt])
def test_provider_or_worker_interruption_preserves_accepted_upstream_work(owner, boundaries, monkeypatch, interruption):
    interrupted = False
    def infer(request):
        nonlocal interrupted
        turn = native_script(request)
        if selected_call(turn)["name"] == "run_diagnostic" and not interrupted:
            interrupted = True
            raise interruption("Synthetic worker/provider interruption")
        return turn
    monkeypatch.setattr(model_access, "invoke_tools", infer)
    _, _, chat = analysis.create(owner)
    run, _ = analysis.prepare(owner, chat)
    if interruption is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            jobs.process(UUID(run["id"]))
    else:
        jobs.process(UUID(run["id"]))
    failed = analysis.read(owner, run["id"])
    assert failed["status"] == ("running" if interruption is KeyboardInterrupt else "failed")
    assert not boundaries
    handoff = failed["agents"][0]["handoff_hash"]
    if interruption is OSError:
        assert owner.post(f"/api/analyses/{run['id']}/retry").status_code == 202
    jobs.process(UUID(run["id"]))
    saved = analysis.read(owner, run["id"])
    assert saved["status"] == "awaiting_approval", saved["error"]
    assert saved["agents"][0]["handoff_hash"] == handoff
    assert saved["agents"][0]["model_calls"] == failed["agents"][0]["model_calls"]
    with Session(engine()) as db:
        attempts = list(db.scalars(select(AnalysisTaskAttempt).join(AnalysisTask).where(
            AnalysisTaskAttempt.run_id == UUID(run["id"]), AnalysisTask.agent == "design", AnalysisTask.parent_id.is_(None)).order_by(AnalysisTaskAttempt.number)))
        index = next(i for i, item in enumerate(attempts) if item.status in {"failed", "uncertain"})
        assert attempts[index].status == ("uncertain" if interruption is KeyboardInterrupt else "failed")
        # Remaining allowance changes between turns, but the saved conversation is unchanged.
        assert attempts[index].request["contents"][:-1] == attempts[index + 1].request["contents"][:-1]
    assert len(boundaries) == 1 and saved["calls"] == 17
    monkeypatch.setattr(model_access, "invoke_tools", lambda request: pytest.fail("Completed work must replay without a provider call"))
    jobs.process(UUID(run["id"]))
    assert analysis.read(owner, run["id"]) == saved


@pytest.mark.parametrize("reject_count", [1, 2])
def test_malformed_tool_arguments_are_correctable_without_execution(owner, boundaries, monkeypatch, reject_count):
    rejected = 0
    def infer(request):
        nonlocal rejected
        turn = native_script(request)
        if selected_call(turn)["name"] == "run_diagnostic" and rejected < reject_count:
            rejected += 1
            assert not boundaries
            return tool_turn("run_diagnostic", {"action": "invented_action"})
        return turn
    monkeypatch.setattr(model_access, "invoke_tools", infer)
    saved = analysis.prepared(owner)
    assert saved["calls"] == 16 + reject_count and len(boundaries) == 1
    detail = agent_detail(owner, saved, "design")
    bad = [r for session in detail["sessions"] for turn in session["attempts"] for r in turn["tool_receipts"].values() if r.get("complete") and not r["result"]["ok"]]
    assert len(bad) == reject_count and all(r["result"]["error"] == "Correct the tool arguments." for r in bad)


def test_historical_active_run_remains_read_only_without_blocking_fresh_contract(owner, boundaries, monkeypatch):
    _, _, chat = analysis.create(owner)
    current = prompts.VERSION
    monkeypatch.setattr(prompts, "VERSION", "checkpoint-harness-11")
    historical, submitted = analysis.prepare(owner, chat)
    monkeypatch.setattr(prompts, "VERSION", current)
    fresh, _ = analysis.prepare(owner, chat)
    assert fresh["id"] != historical["id"] and not fresh["read_only"]
    old = analysis.read(owner, historical["id"])
    assert old["read_only"] and old["status"] == "queued" and old["calls"] == 0
    jobs.process(UUID(historical["id"]))
    assert analysis.read(owner, historical["id"]) == old
    assert owner.post(f"/api/analyses/{historical['id']}/retry").status_code == 409
    replay = owner.post(f"/api/conversations/{chat['id']}/analyses", json=submitted)
    assert replay.status_code == 202 and replay.json() == old
    duplicate = owner.post(f"/api/conversations/{chat['id']}/analyses", json={**submitted, "submission_id": str(uuid4())})
    assert duplicate.status_code == 409 and "active analysis" in duplicate.json()["detail"]
