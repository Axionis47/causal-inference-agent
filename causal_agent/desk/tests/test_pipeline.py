"""Process failures and failed selected leads retain their authorized identity."""

import subprocess

from causal_agent.common.contracts import Handoff, Scope
from causal_agent.desk import pipeline


def handoff():
    return Handoff(
        family="adjustment",
        specialist="dowhy",
        supported_now=True,
        outcome="y",
        treatment="t",
        scope=Scope(),
        pack_name="case",
        relevant_columns=[],
        chosen_assumption="explicit study account",
        reasons=[],
        context_id="authorized-context",
        dataset_identity="authorized-data",
    )


def test_process_failure_is_a_recorded_execution_failure(tmp_path, monkeypatch):
    path = tmp_path / "handoff.json"
    path.write_text(handoff().model_dump_json())
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 17, "", "engine failed"))
    result = pipeline.run(path, 1, "case", "q")
    assert result.status == "pipeline_error" and result.effect is None
    record = result.specialist_result
    assert record["context_id"] == "authorized-context" and record["dataset_identity"] == "authorized-data"
    assert record["terminal"]["kind"] == "execution_failure"
    assert record["execution_attempts"][0]["exit_code"] == 17
    assert record["feasibility"]["terminal_kind"] == "execution_failure"
    assert (tmp_path / "result.json").is_file()


def test_a_successful_rival_does_not_replace_the_failed_selected_lead():
    result = pipeline.record(
        "case",
        "q",
        1,
        handoff(),
        {
            "status": "partial",
            "lead_design_id": "lead",
            "designs": {"lead": {}, "rival": {}},
            "estimates": [
                {"design_id": "lead", "error": "rank failure", "value": None},
                {"design_id": "rival", "error": None, "value": 4.0},
            ],
        },
        {},
        "",
    )
    assert result.lead_design_id == "lead"
    assert result.effect is None and result.estimator is None
    assert result.specialist_result["estimates"][1]["value"] == 4.0
