"""Per-job execution context bundling the substrate handles every agent needs.

The orchestrator builds one AgentCtx per job at dispatch time and passes
it to every ``agent.run(input, ctx)``. Storage and finalisation read the
ctx back for the snapshots (budget, traces, events, decisions) the
notebook and persistence layers consume.

The ctx is the only object an agent receives besides its frozen
AgentInput. Slots an agent reads from state flow through AgentInput;
ctx carries cross-cutting capabilities only (LLM access, audit log,
budget, traces, SSE bus, decision log, dataset record loader).

Frozen dataclass not Pydantic: the handles bound here are not data,
they are mutable substrate buffers. Pydantic would need
arbitrary_types_allowed and give no validation in return. frozen=True
still prevents accidental rebinding of the handles themselves; the
buffers they point to are mutated through their own public APIs.

The record_loader is a callable, not the record, so agents that never
touch the dataset never trigger the load and the closure keeps
``download_id`` hidden from agent code. ``ctx.record`` resolves the
loader on each access.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.domain.download import DownloadRecord
from src.llm.client import LLMClient

from .budget import BudgetTracker
from .events import DecisionLog, SSEEventBus
from .tool_log import ToolCallLog
from .trace import TraceBuffer

RecordLoader = Callable[[], DownloadRecord]


@dataclass(frozen=True, slots=True)
class AgentCtx:
    """Per-job substrate handles passed to every ``agent.run()``.

    Built once by the orchestrator at job dispatch. The bindings are
    immutable; the buffers and tracker behind them are mutable and own
    their concurrency story (single event loop, no locks).
    """

    llm: LLMClient
    tool_log: ToolCallLog
    budget: BudgetTracker
    trace_buffer: TraceBuffer
    events: SSEEventBus
    decisions: DecisionLog
    record_loader: RecordLoader

    @property
    def record(self) -> DownloadRecord:
        """Resolve the DownloadRecord for this job.

        Calls the loader on every access. Loaders are expected to be
        cheap (a closure over a cached value, typically); callers that
        need to pin a snapshot should bind the result locally.
        """
        return self.record_loader()
