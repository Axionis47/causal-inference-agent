"""Launch the analysis for a confirmed job: the run_analysis hand-off.

Re-registers the input state in the manager's live tables so SSE works,
flips the public status to running_analysis, loads the frame, builds a
fresh run state (or reuses a persisted one), and drives the spine in the
job task. Outcomes map to the public statuses: complete -> completed,
parked -> waiting_for_user, failed/frontier -> failed.
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.core import AnalysisRunState, RunStatus
from src.analysis_v2.persistence import load_run, save_run
from src.analysis_v2.state import AnalysisState, JobStatus

from .graph import run_spine
from .loader import load_analysis_frame
from .registry import build_agents

logger = structlog.get_logger(__name__)


async def start(state: AnalysisState, manager: Any) -> dict[str, Any]:
    """Spawn the analysis task for a CONFIRMED job. Returns the launch ack."""
    if state.status not in (JobStatus.CONFIRMED, JobStatus.WAITING_FOR_USER):
        raise ValueError(
            f"job {state.job_id} is {state.status.value}, not confirmed; cannot run"
        )
    if not (state.causal_question or "").strip():
        raise ValueError(f"job {state.job_id} has no causal question")

    job_id = state.job_id
    state.status = JobStatus.RUNNING_ANALYSIS
    state.touch()
    await manager.firestore.update_job(state)
    await manager.firestore.save_parked_state(state)

    task = asyncio.create_task(_run_job(state, manager))
    async with manager._jobs_lock:
        manager._running_jobs[job_id] = task
    logger.info("analysis_launched", job_id=job_id)
    return {"resumed": True, "status": state.status.value}


async def _run_job(state: AnalysisState, manager: Any) -> None:
    job_id = state.job_id
    manager._active_states[job_id] = state

    def emit(event_type: str, data: dict) -> None:
        state.push_sse_event(event_type, data)

    try:
        run = await _load_or_create_run(state)
        frame = load_analysis_frame(state)
        ctx = AgentCtx(
            job_id=job_id, run=run, frame=frame, input_state=state, emit=emit
        )
        emit(
            "analysis_started",
            {"headline": f"analysis started: {run.causal_question[:120]}"},
        )
        outcome = await run_spine(ctx, build_agents())
        await _finalize(state, run, manager, outcome)
    except asyncio.CancelledError:
        state.mark_cancelled()
        await manager.firestore.update_job(state)
        raise
    except Exception as exc:
        logger.error("analysis_job_failed", job_id=job_id, error=str(exc))
        state.mark_failed(str(exc), "analysis_runner")
        await manager.firestore.update_job(state)
    finally:
        async with manager._jobs_lock:
            manager._running_jobs.pop(job_id, None)
        manager._active_states.pop(job_id, None)


async def _load_or_create_run(state: AnalysisState) -> AnalysisRunState:
    existing = await load_run(state.job_id)
    if existing is not None and existing.status in (
        RunStatus.PENDING,
        RunStatus.RUNNING,
        RunStatus.WAITING_FOR_USER,
    ):
        existing.status = RunStatus.RUNNING
        return existing
    run = AnalysisRunState(
        job_id=state.job_id,
        causal_question=(state.causal_question or "").strip(),
        user_context=state.dataset_info.user_provided_context,
        dataset_name=state.dataset_info.name,
        ignored_columns=list(state.ignored_columns),
        status=RunStatus.RUNNING,
    )
    await save_run(run)
    return run


async def _finalize(
    state: AnalysisState, run: AnalysisRunState, manager: Any, outcome: str
) -> None:
    if outcome == "complete":
        from src.analysis_v2.core import AnalysisStage, GateResult

        if run.current_state != AnalysisStage.S12_JOB_COMPLETE:
            run.record_transition(
                to_state=AnalysisStage.S12_JOB_COMPLETE,
                agent_name=None,
                gate_result=GateResult.advance(),
            )
        run.mark_completed()
        await save_run(run)
        state.status = JobStatus.COMPLETED
        state.completed_at = run.completed_at
        state.touch()
        state.push_sse_event(
            "analysis_completed", {"headline": "analysis complete"}
        )
    elif outcome == "parked":
        state.status = JobStatus.WAITING_FOR_USER
        state.touch()
        state.push_sse_event(
            "analysis_waiting_for_user",
            {"headline": "the plan needs your confirmation"},
        )
        await manager.firestore.save_parked_state(state)
    else:  # failed | frontier
        state.mark_failed(run.error_message or "analysis failed", "analysis_runner")
    await manager.firestore.update_job(state)
    logger.info("analysis_finished", job_id=state.job_id, outcome=outcome)
