"""BudgetTracker: per-job caps at three levels + wall clock."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.analysis.substrate.budget import (
    BudgetExceeded,
    BudgetSnapshot,
    BudgetTracker,
)
from src.domain.job import BudgetSpec


def _fake_clock(start: datetime):
    """Returns (clock, advance) where advance(dt: timedelta) moves time forward."""
    now = [start]

    def clock() -> datetime:
        return now[0]

    def advance(delta: timedelta) -> None:
        now[0] += delta

    return clock, advance


def test_tracker_starts_at_zero():
    t = BudgetTracker(BudgetSpec())
    assert t.agent_loops == 0
    assert t.total_tool_calls == 0
    assert t.llm_tokens == 0
    assert t.tool_calls_for("anything") == 0


def test_record_agent_loop_up_to_cap_does_not_raise():
    t = BudgetTracker(BudgetSpec(max_agent_loops=3))
    for _ in range(3):
        t.record_agent_loop()
    assert t.agent_loops == 3


def test_record_agent_loop_past_cap_raises():
    t = BudgetTracker(BudgetSpec(max_agent_loops=2))
    t.record_agent_loop()
    t.record_agent_loop()
    with pytest.raises(BudgetExceeded) as exc:
        t.record_agent_loop()
    assert exc.value.cap == "max_agent_loops"
    assert exc.value.current == 3
    assert exc.value.limit == 2


def test_per_agent_tool_call_caps_are_independent():
    t = BudgetTracker(BudgetSpec(max_tool_calls_per_agent=2))
    t.record_tool_call("profiler")
    t.record_tool_call("profiler")
    t.record_tool_call("eda")
    t.record_tool_call("eda")
    assert t.tool_calls_for("profiler") == 2
    assert t.tool_calls_for("eda") == 2


def test_per_agent_tool_call_cap_trips():
    t = BudgetTracker(BudgetSpec(max_tool_calls_per_agent=2))
    t.record_tool_call("profiler")
    t.record_tool_call("profiler")
    with pytest.raises(BudgetExceeded) as exc:
        t.record_tool_call("profiler")
    assert exc.value.cap == "max_tool_calls_per_agent"


def test_total_tool_call_cap_trips_across_agents():
    t = BudgetTracker(BudgetSpec(max_total_tool_calls=3))
    t.record_tool_call("a")
    t.record_tool_call("b")
    t.record_tool_call("c")
    with pytest.raises(BudgetExceeded) as exc:
        t.record_tool_call("a")
    assert exc.value.cap == "max_total_tool_calls"


def test_llm_token_cap_trips():
    t = BudgetTracker(BudgetSpec(max_llm_tokens=1000))
    t.record_llm_tokens(500)
    t.record_llm_tokens(500)
    with pytest.raises(BudgetExceeded) as exc:
        t.record_llm_tokens(1)
    assert exc.value.cap == "max_llm_tokens"
    assert exc.value.limit == 1000


def test_negative_llm_tokens_raise_value_error():
    t = BudgetTracker(BudgetSpec())
    with pytest.raises(ValueError, match="non-negative"):
        t.record_llm_tokens(-1)


def test_wall_clock_does_not_trip_before_limit():
    clock, advance = _fake_clock(datetime(2025, 1, 1, tzinfo=timezone.utc))
    t = BudgetTracker(BudgetSpec(wall_clock_seconds=60), clock=clock)
    advance(timedelta(seconds=59))
    t.record_agent_loop()  # 59s elapsed, still within limit


def test_wall_clock_trips_on_next_record():
    clock, advance = _fake_clock(datetime(2025, 1, 1, tzinfo=timezone.utc))
    t = BudgetTracker(BudgetSpec(wall_clock_seconds=60), clock=clock)
    advance(timedelta(seconds=30))
    t.record_agent_loop()
    advance(timedelta(seconds=31))  # total 61s, past limit
    with pytest.raises(BudgetExceeded) as exc:
        t.record_agent_loop()
    assert exc.value.cap == "wall_clock_seconds"
    assert exc.value.limit == 60


def test_wall_clock_checked_on_tool_call_too():
    clock, advance = _fake_clock(datetime(2025, 1, 1, tzinfo=timezone.utc))
    t = BudgetTracker(BudgetSpec(wall_clock_seconds=30), clock=clock)
    advance(timedelta(seconds=31))
    with pytest.raises(BudgetExceeded) as exc:
        t.record_tool_call("agent")
    assert exc.value.cap == "wall_clock_seconds"


def test_elapsed_seconds_uses_injected_clock():
    clock, advance = _fake_clock(datetime(2025, 1, 1, tzinfo=timezone.utc))
    t = BudgetTracker(BudgetSpec(), clock=clock)
    assert t.elapsed_seconds() == 0
    advance(timedelta(seconds=42))
    assert t.elapsed_seconds() == 42


def test_snapshot_round_trip():
    t = BudgetTracker(BudgetSpec())
    t.record_agent_loop()
    t.record_tool_call("profiler")
    t.record_tool_call("eda")
    t.record_tool_call("eda")
    t.record_llm_tokens(123)

    snap = t.snapshot()
    assert snap.agent_loops == 1
    assert snap.total_tool_calls == 3
    assert snap.tool_calls_by_agent == {"profiler": 1, "eda": 2}
    assert snap.llm_tokens == 123

    parsed = BudgetSnapshot.model_validate(snap.model_dump())
    assert parsed == snap


def test_snapshot_after_partial_progress_is_consistent():
    t = BudgetTracker(BudgetSpec(max_agent_loops=2))
    t.record_agent_loop()
    t.record_agent_loop()
    with pytest.raises(BudgetExceeded):
        t.record_agent_loop()
    # Snapshot still reflects the over-cap counter so traces can show
    # exactly where the trip happened.
    snap = t.snapshot()
    assert snap.agent_loops == 3
