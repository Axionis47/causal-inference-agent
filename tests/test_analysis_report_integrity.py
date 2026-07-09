"""Explanation cannot amend frozen assumptions after numerical execution."""

import json

import test_analysis as analysis
from test_agents_journey import agent_detail, native_script, selected_call

from backend import model_access

storage = analysis.storage
boundaries = analysis.boundaries


def test_reporting_cannot_register_a_new_post_result_assumption(owner, boundaries, monkeypatch):
    def infer(request):
        turn = native_script(request)
        if selected_call(turn)["name"] == "set_report":
            selected_call(turn)["args"]["assumptions"] = [{"id": "after_results", "statement": "Invented after execution"}]
        return turn
    monkeypatch.setattr(model_access, "invoke_tools", infer)
    prepared = analysis.prepared(owner)
    saved = analysis.approval(owner, prepared["id"])
    assert saved["status"] == "blocked" and saved["snapshot_hash"] == prepared["snapshot_hash"]
    assert saved["programs"][0]["execution"]["valid"]
    detail = agent_detail(owner, saved, "report")
    assert len(detail["sessions"]) == 1, "A rejected draft must not reach the independent reviewer."
    responses = [r["result"] for turn in detail["sessions"][0]["attempts"] for r in turn["tool_receipts"].values() if r.get("complete") and r["name"] == "set_report"]
    assert responses and all(not r["ok"] and any(i["type"] == "extra_forbidden" for i in r["issues"]) for r in responses)


def test_reporting_can_cite_frozen_assumptions_without_reauthoring_them(owner, boundaries, monkeypatch):
    reused = []
    def infer(request):
        turn = native_script(request)
        if selected_call(turn)["name"] == "set_report":
            packet = json.loads(json.loads(request.contents_json)[0]["parts"][0]["text"])
            assumption = packet["accepted_handoffs"][0]["records"]["snapshot"]["frame"]["causal"]["assumptions"][0]
            reused.append(assumption["id"])
            selected_call(turn)["args"]["citations"] = ["execution", "assumption:" + assumption["id"]]
        return turn
    monkeypatch.setattr(model_access, "invoke_tools", infer)
    prepared = analysis.prepared(owner)
    saved = analysis.approval(owner, prepared["id"])
    assert saved["status"] == "succeeded", saved["error"]
    conclusion = agent_detail(owner, saved, "report")["handoff"]["records"]["conclusion"]
    assert "assumption:" + reused[0] in conclusion["citations"]
    assert saved["frame"]["causal"]["assumptions"] == prepared["frame"]["causal"]["assumptions"]
    assert saved["snapshot_hash"] == prepared["snapshot_hash"]


def test_reporting_binds_only_exact_cited_question_literals(owner, boundaries, monkeypatch):
    import analysis_context_fixture
    original_revision = analysis_context_fixture.ready_revision
    monkeypatch.setattr(analysis_context_fixture, "ready_revision", lambda *args, **kwargs: original_revision(*args, **kwargs, question="Randomized group effect on score with a 95% confidence interval."))
    inspected = []
    def infer(request):
        turn = native_script(request)
        if selected_call(turn)["name"] == "set_report":
            contents = json.loads(request.contents_json)
            observations = [p["function_response"]["response"] for content in contents for p in content.get("parts", []) if "function_response" in p and p["function_response"]["name"] == "inspect_results"]
            inspected.extend(observations)
            selected_call(turn)["args"]["report"]["text"] += " The requested confidence level is 95%."
            selected_call(turn)["args"]["citations"].append("question")
        return turn
    monkeypatch.setattr(model_access, "invoke_tools", infer)
    prepared = analysis.prepared(owner)
    saved = analysis.approval(owner, prepared["id"])
    assert saved["status"] == "succeeded", saved["error"]
    assert inspected and "question" in inspected[0]["citation_handles"]
    assert 95 in inspected[0]["numeric_evidence"]["question"]["supported_values"]
    assert {"value": 95.0, "source": "question"} in saved["result"]["report"]["numbers"]
