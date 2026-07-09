"""Generated descriptive prose cannot publish unobserved numerical results."""

import json

import test_analysis as analysis
from test_agents_journey import agent_detail, native_script, selected_call

from backend import model_access
from backend.analysis import code_checks

storage = analysis.storage
boundaries = analysis.boundaries


def test_program_description_is_derived_while_raw_provider_prose_remains_inspectable(owner, boundaries, monkeypatch):
    fabricated = "The treatment coefficient is 0.000 with confidence interval [0.000, 0.000]."
    def infer(request):
        turn = native_script(request)
        call = selected_call(turn)
        if call["name"] == "submit_program":
            call["args"]["summary"] = fabricated
            return model_access.ToolTurn(turn.content, json.dumps(turn.content), turn.metrics)
        return turn
    monkeypatch.setattr(model_access, "invoke_tools", infer)
    saved = analysis.prepared(owner)
    expected = code_checks.program_summary(saved["snapshot"]["plan"])
    assert saved["programs"][0]["program"]["summary"] == expected and "0.000" not in expected
    detail = agent_detail(owner, saved, "computation")
    turn = next(turn for session in detail["sessions"] for turn in session["attempts"]
                if any(r["name"] == "submit_program" for r in turn["tool_receipts"].values()))
    assert turn["provider_content"]["parts"][0]["function_call"]["args"]["summary"] == fabricated
    assert saved["programs"][0]["execution"] is None
