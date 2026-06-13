"""FlowAuditResult (S11a): the end-of-run information-flow check.

A deterministic audit that the run is internally consistent before it completes:
the estimate adjusted for exactly what the causal model sanctioned, the claim
strength is consistent with whether the effect is identified, and every signal
the investigator raised is accounted for. A logical inconsistency fails the run;
unaccounted signals are reported, not failed.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class FlowSignalStatus(StrEnum):
    USED = "used"  # consumed by a downstream stage
    WAIVED = "waived"  # available but deliberately not used
    UNACCOUNTED = "unaccounted"  # raised upstream, not reflected downstream


class FlowSignal(BaseModel):
    signal: str = Field(min_length=1, max_length=300)
    status: FlowSignalStatus
    note: str = Field(default="", max_length=300)


class FlowAuditResult(BaseModel):
    passed: bool
    signals: list[FlowSignal] = Field(default_factory=list)
    inconsistencies: list[str] = Field(default_factory=list)
