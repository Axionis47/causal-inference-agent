"""Agent trace recording: per-action diary entries with bounded growth.

A trace is one structured diary entry per agent action: who, when, what
the agent did, the LLM's reasoning, inputs and outputs, tools called,
duration, tokens. Traces are the story of the job for a human reader.
They are not the operator transcript (that is structlog) and not the
tool-call audit (that is ToolCallLog).

Two ways to record:

1. ``record_trace(buffer, **fields)`` is a plain callable. Agents that
   want sub-step granularity, for example the react loop emitting one
   trace per iteration, use this directly.

2. ``traced_agent_call(buffer, agent_name=, inputs=)`` is an async
   context manager. The orchestrator wraps each agent dispatch with it
   so one diary entry is recorded automatically per trigger, including
   an error trace on exception.

Both write into the same TraceBuffer. The buffer owns truncation
(per-trace, on append) and lazy compression (when the list grows past
``2 * limits.max_traces``, old traces are collapsed into one summary
trace so memory stays bounded).
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentTrace(BaseModel):
    """One diary entry for an agent action.

    Mutable on purpose: the buffer truncates oversized fields in place
    after construction, matching the existing AnalysisState behaviour.
    """

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action: str
    reasoning: str | None = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    tools_called: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)


class TraceLimits(BaseModel):
    """Bounds for a TraceBuffer. Defaults match the legacy state values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_traces: int = Field(default=100, ge=1, le=10_000)
    max_output_len: int = Field(default=1000, ge=10, le=100_000)
    max_reasoning_len: int = Field(default=1000, ge=10, le=100_000)


class TraceBuffer:
    """In-memory diary with per-trace truncation and lazy summary compression.

    The buffer is mutated only by whoever holds the job state (single
    event loop, no locks). Lazy compression matches the legacy threshold:
    only when ``len(traces) > 2 * max_traces`` does the buffer collapse
    older traces into one summary, so the list saws between roughly
    ``max_traces // 2 + 1`` and ``2 * max_traces`` rather than running
    on every append.
    """

    def __init__(self, limits: TraceLimits | None = None) -> None:
        self._limits = limits or TraceLimits()
        self._traces: list[AgentTrace] = []

    @property
    def limits(self) -> TraceLimits:
        return self._limits

    def __len__(self) -> int:
        return len(self._traces)

    def view(self) -> list[AgentTrace]:
        """Read-only snapshot of the current buffer contents."""
        return list(self._traces)

    def append(self, trace: AgentTrace) -> None:
        self._truncate(trace)
        self._traces.append(trace)
        if len(self._traces) > 2 * self._limits.max_traces:
            self._compress()

    def _truncate(self, trace: AgentTrace) -> None:
        cap = self._limits.max_output_len
        if trace.outputs:
            truncated: dict[str, Any] = {}
            for k, v in trace.outputs.items():
                str_v = str(v)
                if len(str_v) > cap:
                    truncated[k] = str_v[:cap] + "...[truncated]"
                else:
                    truncated[k] = v
            trace.outputs = truncated

        rcap = self._limits.max_reasoning_len
        if trace.reasoning and len(trace.reasoning) > rcap:
            trace.reasoning = trace.reasoning[:rcap] + "...[truncated]"

    def _compress(self) -> None:
        if len(self._traces) <= self._limits.max_traces:
            return
        keep = self._limits.max_traces // 2
        old = self._traces[:-keep]
        recent = self._traces[-keep:]
        self._traces = [_summarise(old)] + recent


def _summarise(traces: list[AgentTrace]) -> AgentTrace:
    """Collapse a batch of traces into one summary trace.

    Per agent: action count, total duration, union of tools, last 200
    chars of reasoning. Tokens aggregate across two historic key shapes
    (``input_tokens``/``output_tokens`` and legacy ``input``/``output``)
    so older persisted traces are not dropped.
    """
    agent_info: dict[str, dict[str, Any]] = {}
    total_duration = 0
    total_tokens = {"input": 0, "output": 0}

    for tr in traces:
        info = agent_info.setdefault(
            tr.agent_name,
            {"actions": [], "duration_ms": 0, "tools": set(), "last_reasoning": ""},
        )
        info["actions"].append(tr.action)
        info["duration_ms"] += tr.duration_ms
        if tr.tools_called:
            info["tools"].update(tr.tools_called)
        if tr.reasoning:
            info["last_reasoning"] = tr.reasoning[:200]

        total_duration += tr.duration_ms
        total_tokens["input"] += tr.token_usage.get(
            "input_tokens", tr.token_usage.get("input", 0)
        )
        total_tokens["output"] += tr.token_usage.get(
            "output_tokens", tr.token_usage.get("output", 0)
        )

    summary_text = f"Summary of {len(traces)} compressed traces:\n"
    for agent, info in agent_info.items():
        tools_str = f", tools={list(info['tools'])}" if info["tools"] else ""
        summary_text += (
            f"  {agent}: {len(info['actions'])} actions, "
            f"{info['duration_ms']}ms{tools_str}\n"
        )

    return AgentTrace(
        agent_name="trace_summary",
        action="compressed_history",
        reasoning=summary_text,
        outputs={
            "trace_count": len(traces),
            "agents": {
                a: {
                    "n_actions": len(info["actions"]),
                    "duration_ms": info["duration_ms"],
                    "tools_used": sorted(info["tools"]),
                    "last_reasoning_preview": info["last_reasoning"],
                }
                for a, info in agent_info.items()
            },
            "total_duration_ms": total_duration,
            "compressed_at": datetime.now(timezone.utc).isoformat(),
        },
        duration_ms=0,
        token_usage=total_tokens,
    )


def record_trace(buffer: TraceBuffer, **fields: Any) -> AgentTrace:
    """Build an AgentTrace from kwargs and append it to the buffer.

    Returned for chaining; the buffer holds the canonical reference.
    """
    trace = AgentTrace(**fields)
    buffer.append(trace)
    return trace


@dataclass
class TraceHandle:
    """Mutable slot filled in by the orchestrator during a wrapped dispatch.

    After the agent returns, the orchestrator sets outputs/reasoning/etc.
    on the handle. ``traced_agent_call`` reads the handle and records
    one trace when the context exits.
    """

    agent_name: str
    inputs: dict[str, Any]
    outputs: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    tools_called: list[str] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)


@asynccontextmanager
async def traced_agent_call(
    buffer: TraceBuffer,
    *,
    agent_name: str,
    inputs: dict[str, Any] | None = None,
) -> AsyncIterator[TraceHandle]:
    """Wrap an agent dispatch and record exactly one trace at exit.

    On success: ``action="execute"`` with whatever the caller wrote on
    the handle. On exception: ``action="execution_failed"`` with the
    error type and message in outputs, then the exception re-raises so
    the orchestrator's retry / criticality policy decides what happens.
    """
    handle = TraceHandle(agent_name=agent_name, inputs=dict(inputs or {}))
    start = time.perf_counter()
    try:
        yield handle
    except Exception as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        record_trace(
            buffer,
            agent_name=agent_name,
            action="execution_failed",
            inputs=handle.inputs,
            outputs={"error": str(exc), "error_type": type(exc).__name__},
            duration_ms=duration_ms,
        )
        raise
    duration_ms = int((time.perf_counter() - start) * 1000)
    record_trace(
        buffer,
        agent_name=agent_name,
        action="execute",
        inputs=handle.inputs,
        outputs=handle.outputs,
        reasoning=handle.reasoning,
        tools_called=handle.tools_called,
        token_usage=handle.token_usage,
        duration_ms=duration_ms,
    )
