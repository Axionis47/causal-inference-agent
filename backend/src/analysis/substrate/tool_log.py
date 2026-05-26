"""Tool-call log with args_hash dedupe and loop detection.

Every tool invocation by an agent flows through a ToolCallLog. The log
serves three jobs:

1. Audit: every call (including deduped ones) is recorded with its
   normalised args hash, so traces can show what the agent actually
   did vs. asked to do.
2. Dedupe: the dispatcher can ask the log whether (tool, args_hash)
   was seen before in this job. Pure tools return the cached summary;
   state-changing tools surface a forced-break message to the LLM.
3. Loop detection: when (tool, args_hash) is repeated CIRCULAR_THRESHOLD
   times across the job, CircularToolUse is raised. The retry envelope
   treats this as non-retryable.

Single event loop, no locks. The log lives on AnalysisState and is
mutated only by the agent that holds the state.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CIRCULAR_THRESHOLD = 5

ToolCallStatus = Literal[
    "pending",
    "ok",
    "error",
    "deduped",
    "forced_break",
]


def args_hash(args: dict[str, Any]) -> str:
    """Stable 16-char hex of canonical-JSON-encoded args.

    Sorted keys, compact separators. Non-JSON values fall back to a
    deterministic representation (`type:repr-truncated`) so identical
    calls hash identically without crashing on numpy / pandas / etc.
    """
    normalised = _normalise(args)
    payload = json.dumps(normalised, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalise(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, set):
        return sorted(_normalise(v) for v in value)
    return f"<{type(value).__name__}:{repr(value)[:200]}>"


class ToolCallRecord(BaseModel):
    """One row in the tool-call audit log."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    agent: str
    tool: str
    args_hash: str
    started_at: datetime
    finished_at: datetime | None = None
    status: ToolCallStatus = "pending"
    result_summary: str = ""
    error: str = ""


class CircularToolUse(Exception):
    """Raised when (tool, args_hash) is invoked >= CIRCULAR_THRESHOLD times."""

    def __init__(self, tool: str, count: int) -> None:
        super().__init__(
            f"Circular tool use: {tool} called {count} times with identical args"
        )
        self.tool = tool
        self.count = count


class ToolCallLog:
    """Append-only record of every tool invocation in a job."""

    def __init__(self) -> None:
        self._records: list[ToolCallRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def begin(self, *, agent: str, tool: str, args: dict[str, Any]) -> ToolCallRecord:
        """Open a new record. Caller must call finish() to close it."""
        h = args_hash(args)
        count = self.count_repeat(tool=tool, h=h)
        if count + 1 >= CIRCULAR_THRESHOLD:
            raise CircularToolUse(tool=tool, count=count + 1)
        rec = ToolCallRecord(
            call_id=uuid.uuid4().hex[:12],
            agent=agent,
            tool=tool,
            args_hash=h,
            started_at=datetime.now(timezone.utc),
        )
        self._records.append(rec)
        return rec

    def finish(
        self,
        record: ToolCallRecord,
        *,
        status: ToolCallStatus,
        result_summary: str = "",
        error: str = "",
    ) -> None:
        record.finished_at = datetime.now(timezone.utc)
        record.status = status
        record.result_summary = result_summary
        record.error = error

    def lookup(self, *, tool: str, h: str) -> ToolCallRecord | None:
        """Most recent successful call with this (tool, args_hash), or None."""
        for r in reversed(self._records):
            if r.tool == tool and r.args_hash == h and r.status == "ok":
                return r
        return None

    def count_repeat(self, *, tool: str, h: str) -> int:
        """How many records exist with this (tool, args_hash), any status."""
        return sum(1 for r in self._records if r.tool == tool and r.args_hash == h)

    def tail(self, n: int = 20) -> list[ToolCallRecord]:
        return self._records[-n:]
