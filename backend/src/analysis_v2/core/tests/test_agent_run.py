"""AgentRun lifecycle and token accounting."""
from __future__ import annotations

import pytest

from src.analysis_v2.core import (
    AgentRun,
    AgentRunStatus,
    AnalysisStage,
    TokenUsage,
    ToolCallRecord,
)


def _run() -> AgentRun:
    return AgentRun(agent="intake", stage=AnalysisStage.S1_INTAKE_PARSED)


def test_finish_rejects_non_terminal_statuses():
    run = _run()
    run.start()
    for bad in (AgentRunStatus.WAITING, AgentRunStatus.RUNNING):
        with pytest.raises(ValueError):
            run.finish(bad)
    run.finish(AgentRunStatus.PASSED)
    assert run.status == AgentRunStatus.PASSED
    assert run.finished_at is not None


def test_elapsed_is_none_before_start_and_positive_after_finish():
    run = _run()
    assert run.elapsed_seconds is None
    run.start()
    run.finish(AgentRunStatus.WARNING)
    assert run.elapsed_seconds is not None
    assert run.elapsed_seconds >= 0.0


def test_token_usage_add_sums_both_directions():
    a = TokenUsage(input_tokens=100, output_tokens=20)
    b = TokenUsage(input_tokens=7, output_tokens=3)
    summed = a.add(b)
    assert (summed.input_tokens, summed.output_tokens) == (107, 23)
    assert summed.total == 130
    # add() returns a new value; operands stay unchanged
    assert a.total == 120 and b.total == 10


def test_run_carries_tool_calls_and_warnings_for_the_tile():
    run = _run()
    run.tool_calls.append(
        ToolCallRecord(name="list_columns", args_summary="{}", ok=True, duration_ms=12)
    )
    run.tool_calls.append(
        ToolCallRecord(name="read_schema", ok=False, error="file missing")
    )
    run.warnings.append("question is ambiguous between did and before_after")
    dumped = run.model_dump(mode="json")
    assert dumped["tool_calls"][1]["error"] == "file missing"
    assert dumped["warnings"] == ["question is ambiguous between did and before_after"]
