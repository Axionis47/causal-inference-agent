"""The data-review gate.

The slice has exactly one gate: after the dataset is downloaded and profiled,
the job is parked for human review of the data + inputs. The old DAG and results
gates are gone with the analysis engine, so the gate logic collapses to parking
and a single "is this parked at the data gate" check.
"""
from __future__ import annotations

from typing import Awaitable, Callable

from src.analysis_v2.state import AnalysisState, JobStatus

StatusCallback = Callable[[AnalysisState], Awaitable[None]]


async def park_for_approval(
    state: AnalysisState,
    status_callback: StatusCallback | None = None,
) -> AnalysisState:
    """Park the job at the data-review gate and persist via the callback.

    Sets AWAITING_APPROVAL; the worker writes the parked state and exits. The job
    then waits for confirm (or reject) to move it on. Analysis is not started.
    """
    state.status = JobStatus.AWAITING_APPROVAL
    state.touch()
    if status_callback is not None:
        await status_callback(state)
    return state


def is_at_data_gate(state: AnalysisState) -> bool:
    """True when the job is parked at the data-review gate, awaiting confirm."""
    return state.status == JobStatus.AWAITING_APPROVAL and not state.is_approved()
