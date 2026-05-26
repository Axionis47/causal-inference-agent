"""AgentCtx: per-job substrate handles bound for an agent.run call."""
from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from src.analysis.substrate.budget import BudgetTracker
from src.analysis.substrate.context import AgentCtx
from src.analysis.substrate.events import DecisionLog, SSEEventBus
from src.analysis.substrate.tool_log import ToolCallLog
from src.analysis.substrate.trace import TraceBuffer
from src.domain.download import DownloadRecord, DownloadStatus
from src.domain.job import BudgetSpec


def _record() -> DownloadRecord:
    return DownloadRecord(
        download_id="d1",
        profile_id="default",
        kaggle_url="https://kaggle.com/owner/slug",
        dataset_ref="owner/slug",
        status=DownloadStatus.COMPLETE,
        created_at=datetime.now(timezone.utc),
    )


class _StubLLM:
    async def generate(self, prompt, system_instruction=None, tools=None):
        return {}

    async def generate_with_function_calling(
        self, prompt, system_instruction, tools, max_iterations=5
    ):
        return {}

    async def generate_structured(
        self, prompt, response_schema, system_instruction=None
    ):
        return response_schema()


def _build_ctx(*, record_loader=None) -> AgentCtx:
    rec = _record()
    return AgentCtx(
        llm=_StubLLM(),
        tool_log=ToolCallLog(),
        budget=BudgetTracker(BudgetSpec()),
        trace_buffer=TraceBuffer(),
        events=SSEEventBus(),
        decisions=DecisionLog(),
        record_loader=record_loader or (lambda: rec),
    )


def test_ctx_exposes_all_substrate_handles():
    ctx = _build_ctx()
    assert ctx.llm is not None
    assert isinstance(ctx.tool_log, ToolCallLog)
    assert isinstance(ctx.budget, BudgetTracker)
    assert isinstance(ctx.trace_buffer, TraceBuffer)
    assert isinstance(ctx.events, SSEEventBus)
    assert isinstance(ctx.decisions, DecisionLog)


def test_ctx_is_frozen():
    ctx = _build_ctx()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.budget = BudgetTracker(BudgetSpec())  # type: ignore[misc]


def test_record_property_invokes_loader_on_each_access():
    calls = {"n": 0}

    def loader() -> DownloadRecord:
        calls["n"] += 1
        return _record()

    ctx = _build_ctx(record_loader=loader)
    _ = ctx.record
    _ = ctx.record
    assert calls["n"] == 2


def test_record_property_returns_loader_value():
    rec = _record()
    ctx = _build_ctx(record_loader=lambda: rec)
    assert ctx.record is rec


def test_handles_are_shared_references_not_copies():
    # Buffers must be the exact objects the orchestrator holds; otherwise
    # writes through ctx would not be visible at finalisation time.
    bus = SSEEventBus()
    decisions = DecisionLog()
    log = ToolCallLog()
    trace_buffer = TraceBuffer()
    budget = BudgetTracker(BudgetSpec())

    ctx = AgentCtx(
        llm=_StubLLM(),
        tool_log=log,
        budget=budget,
        trace_buffer=trace_buffer,
        events=bus,
        decisions=decisions,
        record_loader=_record,
    )

    assert ctx.events is bus
    assert ctx.decisions is decisions
    assert ctx.tool_log is log
    assert ctx.trace_buffer is trace_buffer
    assert ctx.budget is budget
