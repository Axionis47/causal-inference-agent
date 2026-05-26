"""record_trace + traced_agent_call: the two surfaces agents use to write traces."""
from __future__ import annotations

import pytest

from src.analysis.substrate.trace import (
    TraceBuffer,
    record_trace,
    traced_agent_call,
)


def test_record_trace_builds_and_appends_in_one_call():
    buf = TraceBuffer()
    trace = record_trace(
        buf,
        agent_name="profiler",
        action="finalize",
        reasoning="picked treatment",
        tools_called=["finalize_profile"],
    )
    assert len(buf) == 1
    assert buf.view()[0] is trace
    assert trace.agent_name == "profiler"
    assert trace.tools_called == ["finalize_profile"]


def test_record_trace_returns_the_appended_trace():
    buf = TraceBuffer()
    trace = record_trace(buf, agent_name="a", action="x")
    assert buf.view()[0] is trace


@pytest.mark.asyncio
async def test_traced_agent_call_records_one_trace_on_success():
    buf = TraceBuffer()
    async with traced_agent_call(buf, agent_name="profiler", inputs={"job": "j1"}) as h:
        h.outputs = {"profile_complete": True}
        h.reasoning = "treatment is binary"
        h.tools_called = ["get_overview"]

    assert len(buf) == 1
    trace = buf.view()[0]
    assert trace.agent_name == "profiler"
    assert trace.action == "execute"
    assert trace.inputs == {"job": "j1"}
    assert trace.outputs == {"profile_complete": True}
    assert trace.reasoning == "treatment is binary"
    assert trace.tools_called == ["get_overview"]
    assert trace.duration_ms >= 0


@pytest.mark.asyncio
async def test_traced_agent_call_records_failure_trace_and_reraises():
    buf = TraceBuffer()
    with pytest.raises(ValueError, match="boom"):
        async with traced_agent_call(buf, agent_name="eda", inputs={"job": "j2"}):
            raise ValueError("boom")

    assert len(buf) == 1
    trace = buf.view()[0]
    assert trace.agent_name == "eda"
    assert trace.action == "execution_failed"
    assert trace.outputs == {"error": "boom", "error_type": "ValueError"}
    assert trace.inputs == {"job": "j2"}


@pytest.mark.asyncio
async def test_traced_agent_call_inputs_are_copied_not_aliased():
    buf = TraceBuffer()
    inputs = {"job": "j1"}
    async with traced_agent_call(buf, agent_name="a", inputs=inputs):
        inputs["job"] = "mutated_outside"

    assert buf.view()[0].inputs == {"job": "j1"}


@pytest.mark.asyncio
async def test_traced_agent_call_with_no_inputs_records_empty_dict():
    buf = TraceBuffer()
    async with traced_agent_call(buf, agent_name="a"):
        pass
    assert buf.view()[0].inputs == {}


@pytest.mark.asyncio
async def test_traced_agent_call_failure_path_preserves_handle_inputs():
    buf = TraceBuffer()
    with pytest.raises(RuntimeError):
        async with traced_agent_call(buf, agent_name="a", inputs={"k": 1}) as h:
            h.outputs = {"partial": True}  # ignored on failure path
            raise RuntimeError("kaboom")

    trace = buf.view()[0]
    assert trace.inputs == {"k": 1}
    assert "partial" not in trace.outputs


@pytest.mark.asyncio
async def test_traced_agent_call_records_duration():
    import asyncio

    buf = TraceBuffer()
    async with traced_agent_call(buf, agent_name="a"):
        await asyncio.sleep(0.01)

    assert buf.view()[0].duration_ms >= 5
