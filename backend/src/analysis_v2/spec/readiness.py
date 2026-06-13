"""ReadinessResult (S6b): can the frozen plan's lane actually run on this data?

A deterministic gate between the plan freeze (S6) and estimation (S7). It calls
the chosen lane's check_ready (the same preconditions the lane raises on at
runtime) and records the blocking reasons, empty when ready. When the bounded
repair runs, it records which lane it replaced and why.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ReadinessResult(BaseModel):
    lane: str = Field(min_length=1)
    ready: bool
    blocking_reasons: list[str] = Field(default_factory=list)
    repaired_from: str | None = None
    repair_notes: list[str] = Field(default_factory=list)
