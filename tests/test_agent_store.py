import json
from copy import deepcopy
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session
from test_analysis_tasks import run_id as run_id

from backend import model_access
from backend.analysis.agent_store import AgentLimit, HandoffConflict, Store
from backend.db import engine
from backend.models import AnalysisRun

PACKET = {"frozen_inputs": {"version": "fixed-version", "context": "fixed-context"}, "handoffs": []}
REQUEST = model_access.PreparedToolRequest("test", "Test", "[]", "[]")


def turn(name="inspect"):
    content = {"role": "model", "parts": [{"function_call": {"name": name, "args": {}}}]}
    return model_access.ToolTurn(content, json.dumps(content), {"total_tokens": 12})


def test_native_response_and_completed_tools_recover_without_repeat(run_id):
    with Session(engine()) as db:
        store = Store(db, run_id)
        task = store.open("study", PACKET)
        attempt = store.reserve(task, REQUEST)
        store.settle(attempt, turn())
        observed = []
        def execute(key):
            observed.append(key)
            task.draft = {"saved": "processed evidence"}
            return {"value": 4}
        assert store.tool(task, attempt, 0, "inspect", {}, execute) == {"value": 4}
        assert store.tool(task, attempt, 0, "inspect", {}, execute) == {"value": 4}
        assert len(observed) == 1
        assert db.get(AnalysisRun, run_id).calls == 1
        with pytest.raises(HandoffConflict, match="different arguments"):
            store.tool(task, attempt, 0, "inspect", {"changed": True}, execute)
    with Session(engine()) as db:
        store = Store(db, run_id)
        reopened = store.open("study", PACKET)
        assert reopened.draft == {"saved": "processed evidence"}
        saved = store.turns(reopened)[0]
        assert saved.provider_content == turn().content
        assert saved.tool_receipts["0"]["complete"]


def test_failed_and_uncertain_calls_stay_charged_with_parent_budget(run_id):
    with Session(engine()) as db:
        store = Store(db, run_id)
        parent = store.open("study", PACKET)
        child = store.open("study", PACKET, key="study:review", parent=parent, limit=2)
        first = store.reserve(child, REQUEST)
        store.settle(first, None, RuntimeError("provider interrupted"))
        store.reserve(child, REQUEST)
        store.recover()
        assert store.counts("study") == 2
        assert [item.status for item in store.turns(child)] == ["failed", "uncertain"]
        with pytest.raises(AgentLimit):
            store.reserve(child, REQUEST)
        assert db.get(AnalysisRun, run_id).calls == 2


def test_handoff_seals_dependencies_and_rejects_cross_run_or_changed_inputs(run_id):
    with Session(engine()) as db:
        store = Store(db, run_id)
        task = store.open("study", PACKET)
        attempt = store.reserve(task, REQUEST)
        store.settle(attempt, turn("publish"))
        result = store.tool(task, attempt, 0, "publish", {}, lambda key: store.publish(task, {"meaning": "exact accepted fact"}, "Study accepted", []))
        received = store.open("design", {**PACKET, "handoffs": [result]})
        assert store.handoff(task.id, expected_hash=result["hash"])["records"] == {"meaning": "exact accepted fact"}
        assert received.dependency_receipt_ids == [str(task.id)]
        with pytest.raises(HandoffConflict, match="replaced"):
            store.publish(task, {"meaning": "changed"}, "Study accepted", [])
        with pytest.raises(HandoffConflict, match="different inputs"):
            store.open("study", {**PACKET, "question": "different"})
        with pytest.raises(HandoffConflict):
            Store(db, uuid4()).handoff(task.id)
        with pytest.raises(HandoffConflict, match="hash"):
            store.handoff(task.id, expected_hash="0" * 64)


def test_handoff_submission_limit_is_durable_and_idempotent(run_id):
    with Session(engine()) as db:
        store = Store(db, run_id)
        task = store.open("study", deepcopy(PACKET))
        for number in range(3):
            store.submission(task, str(number))
            store.submission(task, str(number))
        assert task.submissions == 3
        with pytest.raises(AgentLimit):
            store.submission(task, "fourth")


def test_three_inflight_model_calls_are_enforced_per_run(run_id):
    with Session(engine()) as db:
        store = Store(db, run_id)
        parent = store.open("design", PACKET)
        for index in range(3):
            child = store.open("design", PACKET, key=f"design:read:{index}", parent=parent, limit=2)
            store.reserve(child, REQUEST)
        with pytest.raises(AgentLimit, match="Three model calls"):
            store.reserve(parent, REQUEST)
