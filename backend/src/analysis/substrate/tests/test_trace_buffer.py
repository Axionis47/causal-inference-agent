"""TraceBuffer: append, truncate, lazy compression, summary shape."""
from __future__ import annotations

from src.analysis.substrate.trace import (
    AgentTrace,
    TraceBuffer,
    TraceLimits,
)


def _trace(**overrides) -> AgentTrace:
    base = {"agent_name": "a", "action": "x"}
    base.update(overrides)
    return AgentTrace(**base)


def test_append_records_trace_and_increments_length():
    buf = TraceBuffer()
    buf.append(_trace())
    assert len(buf) == 1
    assert buf.view()[0].action == "x"


def test_view_returns_a_copy_not_internal_list():
    buf = TraceBuffer()
    buf.append(_trace())
    snapshot = buf.view()
    snapshot.clear()
    assert len(buf) == 1


def test_truncate_long_output_replaces_value_with_truncated_string():
    buf = TraceBuffer(TraceLimits(max_traces=10, max_output_len=100))
    long_val = "x" * 500
    buf.append(_trace(outputs={"big": long_val}))
    stored = buf.view()[0].outputs["big"]
    assert isinstance(stored, str)
    assert stored.endswith("...[truncated]")
    assert len(stored) == 100 + len("...[truncated]")


def test_truncate_keeps_original_value_when_under_cap():
    # Mixed-type preservation: truncated entries are strings, untruncated
    # entries keep their original type. This is the legacy contract.
    buf = TraceBuffer(TraceLimits(max_traces=10, max_output_len=100))
    buf.append(_trace(outputs={"small": [1, 2, 3], "big": "x" * 500}))
    stored = buf.view()[0].outputs
    assert stored["small"] == [1, 2, 3]
    assert isinstance(stored["big"], str)


def test_truncate_long_reasoning():
    buf = TraceBuffer(TraceLimits(max_traces=10, max_output_len=1000, max_reasoning_len=50))
    buf.append(_trace(reasoning="r" * 200))
    assert buf.view()[0].reasoning == "r" * 50 + "...[truncated]"


def test_short_reasoning_is_not_modified():
    buf = TraceBuffer(TraceLimits(max_traces=10, max_output_len=1000, max_reasoning_len=50))
    buf.append(_trace(reasoning="short"))
    assert buf.view()[0].reasoning == "short"


def test_compression_does_not_run_below_2x_threshold():
    # max_traces=4 -> compression should not run until len > 8.
    buf = TraceBuffer(TraceLimits(max_traces=4, max_output_len=1000, max_reasoning_len=1000))
    for i in range(8):
        buf.append(_trace(action=f"a{i}"))
    assert len(buf) == 8
    # No summary entry yet.
    assert all(t.agent_name != "trace_summary" for t in buf.view())


def test_compression_runs_when_list_exceeds_2x_threshold():
    buf = TraceBuffer(TraceLimits(max_traces=4, max_output_len=1000, max_reasoning_len=1000))
    for i in range(9):
        buf.append(_trace(agent_name="profiler", action=f"a{i}"))
    # max_traces=4, keep_count = 4 // 2 = 2 -> [summary, recent, recent]
    view = buf.view()
    assert len(view) == 3
    assert view[0].agent_name == "trace_summary"
    assert view[0].action == "compressed_history"
    assert view[1].action == "a7"
    assert view[2].action == "a8"


def test_summary_groups_actions_by_agent():
    buf = TraceBuffer(TraceLimits(max_traces=2, max_output_len=1000, max_reasoning_len=1000))
    # max_traces=2, threshold=4; push 5 to trigger compression.
    buf.append(_trace(agent_name="profiler", action="p1"))
    buf.append(_trace(agent_name="eda", action="e1"))
    buf.append(_trace(agent_name="profiler", action="p2"))
    buf.append(_trace(agent_name="eda", action="e2"))
    buf.append(_trace(agent_name="profiler", action="p3"))

    summary = buf.view()[0]
    assert summary.agent_name == "trace_summary"
    agents = summary.outputs["agents"]
    # Compressed old = first 4 traces (p1, e1, p2, e2). Recent kept = 1.
    assert agents["profiler"]["n_actions"] == 2
    assert agents["eda"]["n_actions"] == 2


def test_summary_unions_tools_called_per_agent():
    buf = TraceBuffer(TraceLimits(max_traces=2, max_output_len=1000, max_reasoning_len=1000))
    buf.append(_trace(agent_name="p", tools_called=["overview"]))
    buf.append(_trace(agent_name="p", tools_called=["overview", "analyze"]))
    buf.append(_trace(agent_name="p", tools_called=["finalize"]))
    buf.append(_trace(agent_name="p"))
    buf.append(_trace(agent_name="p"))

    summary = buf.view()[0]
    assert summary.outputs["agents"]["p"]["tools_used"] == ["analyze", "finalize", "overview"]


def test_summary_aggregates_tokens_with_both_key_conventions():
    # Legacy traces used "input"/"output"; current traces use
    # "input_tokens"/"output_tokens". Summary must sum across both.
    buf = TraceBuffer(TraceLimits(max_traces=2, max_output_len=1000, max_reasoning_len=1000))
    buf.append(_trace(token_usage={"input_tokens": 10, "output_tokens": 5}))
    buf.append(_trace(token_usage={"input": 3, "output": 2}))
    buf.append(_trace(token_usage={"input_tokens": 7}))
    buf.append(_trace())
    buf.append(_trace())

    summary = buf.view()[0]
    assert summary.token_usage == {"input": 20, "output": 7}


def test_summary_takes_last_200_char_reasoning_preview():
    buf = TraceBuffer(TraceLimits(max_traces=2, max_output_len=10_000, max_reasoning_len=10_000))
    buf.append(_trace(agent_name="p", reasoning="first"))
    buf.append(_trace(agent_name="p", reasoning="x" * 500))
    buf.append(_trace(agent_name="p", reasoning="final"))
    buf.append(_trace(agent_name="p"))
    buf.append(_trace(agent_name="p"))

    summary = buf.view()[0]
    # Last reasoning wins, capped at 200 chars.
    preview = summary.outputs["agents"]["p"]["last_reasoning_preview"]
    assert preview == "final"


def test_limits_are_frozen():
    import pytest
    from pydantic import ValidationError

    limits = TraceLimits()
    with pytest.raises(ValidationError):
        limits.max_traces = 50  # type: ignore[misc]
