"""Job lifecycle status: the input slice's stages plus the analysis slice's
coarse public states.

The input slice owns pending..confirmed. After launch the analysis slice
reports only three coarse states here (running_analysis, waiting_for_user,
completed); its fine-grained S0..S12 progression lives on the analysis run
state. COMPLETED's literal value 'completed' is load-bearing: the SSE done
check, GET /results, and the storage terminal sets compare that string.
"""
from __future__ import annotations

from enum import StrEnum


class JobStatus(StrEnum):
    PENDING = "pending"
    FETCHING_DATA = "fetching_data"
    PROFILING = "profiling"
    AWAITING_APPROVAL = "awaiting_approval"  # parked: human reviews data + inputs
    CONFIRMED = "confirmed"  # dataset + inputs accepted; analysis not started
    RUNNING_ANALYSIS = "running_analysis"  # the analysis spine is executing
    WAITING_FOR_USER = "waiting_for_user"  # parked at the plan-confirmation gate
    COMPLETED = "completed"  # analysis finished; results + notebook available
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
