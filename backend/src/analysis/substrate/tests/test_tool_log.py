"""ToolCallLog: append-only audit, lookup, repeat counting, circular detection."""
from __future__ import annotations

import pytest

from src.analysis.substrate.tool_log import (
    CIRCULAR_THRESHOLD,
    CircularToolUse,
    ToolCallLog,
    args_hash,
)


def _begin_and_succeed(log: ToolCallLog, *, agent: str, tool: str, args: dict, result: str = "ok") -> None:
    rec = log.begin(agent=agent, tool=tool, args=args)
    log.finish(rec, status="ok", result_summary=result)


def test_begin_records_pending_then_finish_marks_ok():
    log = ToolCallLog()
    rec = log.begin(agent="profiler", tool="get_overview", args={"x": 1})
    assert rec.status == "pending"
    assert rec.finished_at is None

    log.finish(rec, status="ok", result_summary="n_rows=150")
    assert rec.status == "ok"
    assert rec.finished_at is not None
    assert rec.result_summary == "n_rows=150"


def test_finish_with_error_keeps_record_visible():
    log = ToolCallLog()
    rec = log.begin(agent="eda", tool="check_balance", args={"t": "treat"})
    log.finish(rec, status="error", error="treat is non-binary")
    assert rec.status == "error"
    assert rec.error == "treat is non-binary"
    assert len(log) == 1


def test_lookup_returns_most_recent_ok_call_with_same_args():
    log = ToolCallLog()
    _begin_and_succeed(log, agent="p", tool="overview", args={"c": "x"}, result="first")
    _begin_and_succeed(log, agent="p", tool="overview", args={"c": "x"}, result="second")

    hit = log.lookup(tool="overview", h=args_hash({"c": "x"}))
    assert hit is not None
    assert hit.result_summary == "second"


def test_lookup_ignores_failed_calls():
    log = ToolCallLog()
    rec = log.begin(agent="p", tool="overview", args={"c": "x"})
    log.finish(rec, status="error", error="boom")

    assert log.lookup(tool="overview", h=args_hash({"c": "x"})) is None


def test_lookup_misses_when_args_differ():
    log = ToolCallLog()
    _begin_and_succeed(log, agent="p", tool="analyze", args={"col": "a"})

    assert log.lookup(tool="analyze", h=args_hash({"col": "b"})) is None


def test_count_repeat_counts_all_statuses():
    log = ToolCallLog()
    rec1 = log.begin(agent="p", tool="overview", args={"x": 1})
    log.finish(rec1, status="ok")
    rec2 = log.begin(agent="p", tool="overview", args={"x": 1})
    log.finish(rec2, status="error", error="transient")

    assert log.count_repeat(tool="overview", h=args_hash({"x": 1})) == 2


def test_circular_use_raises_at_threshold():
    log = ToolCallLog()
    # Open and close CIRCULAR_THRESHOLD - 1 records; next begin must raise.
    for _ in range(CIRCULAR_THRESHOLD - 1):
        rec = log.begin(agent="p", tool="loopy", args={"x": 1})
        log.finish(rec, status="ok")

    with pytest.raises(CircularToolUse) as exc_info:
        log.begin(agent="p", tool="loopy", args={"x": 1})

    assert exc_info.value.tool == "loopy"
    assert exc_info.value.count == CIRCULAR_THRESHOLD


def test_circular_threshold_does_not_trip_on_different_args():
    log = ToolCallLog()
    for i in range(CIRCULAR_THRESHOLD + 2):
        rec = log.begin(agent="p", tool="overview", args={"x": i})
        log.finish(rec, status="ok")

    assert len(log) == CIRCULAR_THRESHOLD + 2


def test_tail_returns_last_n_records_in_insertion_order():
    log = ToolCallLog()
    for i in range(10):
        rec = log.begin(agent="p", tool="t", args={"i": i})
        log.finish(rec, status="ok")

    last_three = log.tail(3)
    assert len(last_three) == 3
    assert last_three[0].args_hash == args_hash({"i": 7})
    assert last_three[-1].args_hash == args_hash({"i": 9})


def test_log_is_empty_on_construction():
    log = ToolCallLog()
    assert len(log) == 0
    assert log.tail(5) == []
