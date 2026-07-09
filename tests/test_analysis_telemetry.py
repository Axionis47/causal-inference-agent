"""Native provider observations survive correction and domain normalization."""

import json
from uuid import UUID

import test_analysis as analysis
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_agents_journey import agent_detail, native_script, tool_turn

from backend import model_access
from backend.db import engine
from backend.models import AnalysisTaskAttempt

storage = analysis.storage
boundaries = analysis.boundaries


def test_exact_native_response_and_usage_are_durable_before_tool_dispatch(owner, boundaries, monkeypatch):
    observed = {}
    def infer(request):
        with Session(engine()) as db:
            reserved = list(db.scalars(select(AnalysisTaskAttempt).where(AnalysisTaskAttempt.status == "reserved")))
            assert len(reserved) == 1 and reserved[0].request == request.record()
        turn = native_script(request) if observed else tool_turn("unknown_tool", {})
        turn.content["parts"][0]["thought_signature"] = "opaque-provider-signature"
        raw = " \n" + json.dumps(turn.content, indent=2) + "\n "
        identity = "provider-" + str(len(observed))
        metrics = {"provider_response_id": identity, "input_tokens": 120, "output_tokens": 18, "reasoning_tokens": 22, "total_tokens": 160}
        observed[identity] = (raw, metrics)
        return model_access.ToolTurn(turn.content, raw, metrics)
    monkeypatch.setattr(model_access, "invoke_tools", infer)
    saved = analysis.prepared(owner)
    with Session(engine()) as db:
        attempts = list(db.scalars(select(AnalysisTaskAttempt).where(AnalysisTaskAttempt.run_id == UUID(saved["id"]))))
        assert len(attempts) == len(observed)
        for attempt in attempts:
            raw, metrics = observed[attempt.metrics["provider_response_id"]]
            assert attempt.raw_output == raw and attempt.metrics == metrics
            assert attempt.provider_content["parts"][0]["thought_signature"] == "opaque-provider-signature"
    detail = agent_detail(owner, saved, "study")
    assert not detail["sessions"][0]["attempts"][0]["tool_receipts"]["0"]["result"]["ok"]
    assert saved["status"] == "awaiting_approval" and len(boundaries) == 1
