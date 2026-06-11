"""Job lifecycle status for the download / data-review slice.

Trimmed to the stages this slice owns: acquire the dataset, profile it, park it
at the data-review gate, confirm it, plus the failure / cancellation terminals.
Analysis stages are deliberately absent; the analysis slice owns its own
progression after launch.
"""
from __future__ import annotations

from enum import StrEnum


class JobStatus(StrEnum):
    PENDING = "pending"
    FETCHING_DATA = "fetching_data"
    PROFILING = "profiling"
    AWAITING_APPROVAL = "awaiting_approval"  # parked: human reviews data + inputs
    CONFIRMED = "confirmed"  # dataset + inputs accepted; analysis not started
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
