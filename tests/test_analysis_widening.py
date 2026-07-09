"""Scoped delegation reads the agreed pack without widening or mutating it."""

import json
from uuid import uuid4

import test_analysis as analysis
from historical_analysis import work
from test_agents_journey import agent_detail, native_script, tool_turn

from backend import model_access
from backend.analysis import references
from backend.context.contracts import SourceReference, Statement, empty_content

storage = analysis.storage
boundaries = analysis.boundaries


def test_optional_evidence_subtask_is_isolated_charged_and_linked(owner, boundaries, monkeypatch):
    delegated = False
    def infer(request):
        nonlocal delegated
        packet = json.loads(json.loads(request.contents_json)[0]["parts"][0]["text"])
        if "set_target" in request.functions_json and not delegated:
            delegated = True
            return tool_turn("delegate_evidence", {"objective": "Summarize this existing study record.", "keys": [packet["evidence_records"][0]["key"]]})
        if "submit_findings" in request.functions_json:
            assert "candidate" not in packet and not packet["handoffs"]
            assert len(packet["evidence_records"]) == 1
        return native_script(request)
    monkeypatch.setattr(model_access, "invoke_tools", infer)
    saved = analysis.prepared(owner)
    detail = agent_detail(owner, saved, "study")
    children = [session for session in detail["sessions"] if session["parent_id"]]
    evidence = next(session for session in children if "findings" in session["handoff"]["records"])
    assert {"producer": evidence["id"], "hash": evidence["handoff_hash"]} in detail["handoff"]["dependencies"]
    assert saved["agents"][0]["model_calls"] == 8 and saved["calls"] == 18
    assert saved["context"]["revision_id"] == saved["frozen_context"]["context_revision_id"]


def test_source_handles_round_trip_without_fuzzy_reference_repair():
    context = empty_content(uuid4(), ["renamed", "other"])
    cid = next(iter(context.columns))
    source = SourceReference(kind="message", id=uuid4(), author_id=uuid4())
    context.columns[cid].definition["meaning"] = Statement(
        value="Meaning follows the source, not the header.", source=source
    )
    packet = work.packet(context, {cid}, {}, "Which measurement?")
    visible = {key: value for key, value in packet.items() if not key.startswith("_")}
    encoded, aliases, choices = references.selections(visible)
    evidence = encoded["evidence_records"][0]
    restored = references.translate(
        {"column_ids": encoded["task_columns"], "citations": [evidence["key"]]}, aliases
    )
    assert restored == {"column_ids": [cid], "citations": [f"columns.{cid}.definition.meaning"]}
    assert evidence["key"] in choices["citations"]
    assert references.translate("invented.e999", aliases) == "invented.e999"
