"""SSEEventBus + emit_event: bounded FIFO stream of typed wire events."""
from __future__ import annotations

import pytest

from src.analysis.substrate.events import (
    SSEEvent,
    SSEEventBus,
    emit_event,
)


def test_emit_event_appends_and_returns_typed_event():
    bus = SSEEventBus()
    event = emit_event(bus, "agent_started", {"agent_name": "data_profiler"})

    assert len(bus) == 1
    assert isinstance(event, SSEEvent)
    assert event.event_type == "agent_started"
    assert event.data == {"agent_name": "data_profiler"}
    assert bus.view()[0] is event


def test_view_returns_a_copy_not_internal_list():
    bus = SSEEventBus()
    emit_event(bus, "agent_started", {})
    snapshot = bus.view()
    snapshot.clear()
    assert len(bus) == 1


def test_fifo_trim_keeps_last_n_when_over_cap():
    bus = SSEEventBus(max_events=3)
    for i in range(5):
        emit_event(bus, "agent_completed", {"i": i})

    assert len(bus) == 3
    # Oldest two dropped; we keep i=2,3,4.
    assert [e.data["i"] for e in bus.view()] == [2, 3, 4]


def test_view_after_returns_tail_from_offset():
    # Mirrors JobManager.get_sse_events: state.sse_events[after_index:].
    bus = SSEEventBus()
    for i in range(5):
        emit_event(bus, "agent_started", {"i": i})

    tail = bus.view_after(2)
    assert [e.data["i"] for e in tail] == [2, 3, 4]


def test_view_after_returns_empty_when_offset_past_end():
    bus = SSEEventBus()
    emit_event(bus, "agent_started", {})
    assert bus.view_after(10) == []


def test_view_after_zero_returns_all():
    bus = SSEEventBus()
    for i in range(3):
        emit_event(bus, "agent_started", {"i": i})
    assert len(bus.view_after(0)) == 3


def test_event_timestamp_serialises_to_iso_string_in_json_mode():
    # Wire contract is `{event_type, data, timestamp}` with timestamp as
    # an ISO 8601 string. Internal datetime must dump that way.
    bus = SSEEventBus()
    event = emit_event(bus, "agent_started", {})
    dumped = event.model_dump(mode="json")

    assert isinstance(dumped["timestamp"], str)
    assert "T" in dumped["timestamp"]  # ISO format


def test_preserves_wire_contract_event_names():
    # CLAUDE.md section 5: these names are pinned by front-end + tests.
    # Verify the bus does not rewrite them.
    bus = SSEEventBus()
    for name in (
        "dataset_metadata_started",
        "dataset_metadata_ready",
        "dataset_metadata_failed",
        "dataset_download_started",
        "dataset_download_complete",
        "dataset_load_failed",
        "agent_started",
        "agent_completed",
    ):
        emit_event(bus, name, {})

    seen = [e.event_type for e in bus.view()]
    assert seen == [
        "dataset_metadata_started",
        "dataset_metadata_ready",
        "dataset_metadata_failed",
        "dataset_download_started",
        "dataset_download_complete",
        "dataset_load_failed",
        "agent_started",
        "agent_completed",
    ]


def test_extra_fields_on_event_are_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SSEEvent(event_type="x", data={}, weird_field=True)  # type: ignore[call-arg]


def test_max_events_below_one_rejected():
    with pytest.raises(ValueError, match=">= 1"):
        SSEEventBus(max_events=0)


def test_max_events_property_exposed():
    bus = SSEEventBus(max_events=42)
    assert bus.max_events == 42
