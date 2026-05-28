"""Native tool feedback, correction, replay and database immutability."""

import json

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session
from test_agent_store import PACKET, REQUEST
from test_analysis_tasks import run_id as run_id

from backend import model_access
from backend.analysis.agent_context import NoArguments
from backend.analysis.agent_runtime import Agent, Tool
from backend.analysis.agent_store import Store
from backend.db import engine
from backend.validation import Contract


class Submission(Contract):
    reference: str


def response(name, args, identity):
    content = {"role": "model", "parts": [{"thought_signature": "c2lnbmF0dXJl",
        "function_call": {"name": name, "args": args, "id": identity}}]}
    return model_access.ToolTurn(content, json.dumps(content), {"input_tokens": 10, "output_tokens": 20})


def test_agent_observes_tool_rejection_and_corrects_before_handoff(run_id, monkeypatch):
    with Session(engine()) as db:
        store = Store(db, run_id)
        task = store.open("study", PACKET)
        responses = [response("submit", {"reference": "invented"}, "first"), response("submit", {"reference": "accepted-source"}, "second")]
        requests = []

        def model(request):
            requests.append(json.loads(request.contents_json))
            return responses[len(requests) - 1]

        def submit(args, key):
            store.submission(task, key)
            if args.reference != "accepted-source":
                raise ValueError("Choose the offered source: accepted-source")
            return {"handoff": store.publish(task, {"evidence": args.reference}, "Accepted study", [])}

        monkeypatch.setattr(model_access, "invoke_tools", model)
        Agent(store, task, [Tool("submit", "Submit", Submission, submit)], "Test", "test").graph().invoke({})
        assert task.status == "accepted"
        assert len(requests) == 2 and task.submissions == 2
        assert requests[1][1] == responses[0].content
        feedback = requests[1][2]["parts"][0]["function_response"]
        assert feedback["id"] == "first" and feedback["response"]["ok"] is False
        assert "accepted-source" in feedback["response"]["error"]
        receipt = store.turns(task)[-1].tool_receipts["0"]["result"]["handoff"]
        assert "records" not in receipt  # A pointer, not a second handoff copy.
        assert receipt["hash"] == task.handoff_hash


def test_restarts_execute_saved_provider_tool_without_repeating_model(run_id, monkeypatch):
    with Session(engine()) as db:
        store = Store(db, run_id)
        task = store.open("study", PACKET)
        attempt = store.reserve(task, REQUEST)
        store.settle(attempt, response("finish", {}, "saved-call"))
    observed = []
    with Session(engine()) as db:
        store = Store(db, run_id)
        task = store.open("study", PACKET)
        def finish(args, key):
            observed.append(key)
            return {"handoff": store.publish(task, {"result": "saved work"}, "Ready", [])}
        def unexpected(request):
            raise AssertionError("Recovery must consume the saved provider response first")
        monkeypatch.setattr(model_access, "invoke_tools", unexpected)
        agent = Agent(store, task, [Tool("finish", "Finish", NoArguments, finish)], "Test", "test")
        agent.graph().invoke({})
        agent.graph().invoke({})
        assert len(observed) == 1 and store.counts("study") == 1


def test_database_rejects_accepted_handoff_and_completed_tool_tampering(run_id):
    with Session(engine()) as db:
        store = Store(db, run_id)
        task = store.open("study", PACKET)
        attempt = store.reserve(task, REQUEST)
        store.settle(attempt, response("finish", {}, "call"))
        store.tool(task, attempt, 0, "finish", {}, lambda key: store.publish(task, {"fact": "original"}, "Ready", []))
        task_id, attempt_id = task.id, attempt.id
    with engine().begin() as connection:
        with pytest.raises(DBAPIError, match="immutable"):
            connection.execute(text("UPDATE analysis_tasks SET handoff = '{}'::jsonb WHERE id = :id"), {"id": task_id})
    with engine().begin() as connection:
        with pytest.raises(DBAPIError, match="immutable"):
            connection.execute(text("UPDATE analysis_task_attempts SET tool_receipts = '{}'::jsonb WHERE id = :id"), {"id": attempt_id})
