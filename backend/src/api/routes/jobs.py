"""Job management API routes."""

import asyncio
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from src.agents.base import JobStatus
from src.api.rate_limit import limiter
from src.api.schemas import (
    AgentTracesResponse,
    AnalysisResultsResponse,
    CancelJobResponse,
    CausalGraphResponse,
    CreateJobRequest,
    DeleteJobResponse,
    JobDetailResponse,
    JobListResponse,
    JobResponse,
    JobStatusResponse,
    SensitivityResponse,
    TreatmentEffectResponse,
)
from src.api.utils import (
    build_data_context,
    calculate_method_consensus,
    generate_executive_summary,
)
from src.jobs.manager import get_job_manager
from src.logging_config.structured import get_logger
from src.storage.cleanup import CAUSAL_TEMP_DIR

logger = get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Valid job statuses for query parameter validation
VALID_STATUSES = {s.value for s in JobStatus}


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def create_job(request: Request, body: CreateJobRequest) -> JobResponse:
    """Create a new analysis job.

    Submit a Kaggle dataset URL to start causal inference analysis.
    """
    # slowapi requires 'request' as first param with type starlette.requests.Request
    # FastAPI will inject the Pydantic body via the 'body' parameter
    manager = get_job_manager()

    try:
        job_id = await manager.create_job(
            kaggle_url=body.kaggle_url,
            treatment_variable=body.treatment_variable,
            outcome_variable=body.outcome_variable,
            preferences=body.analysis_preferences,
        )

        # Get the created job
        job = await manager.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create job",
            )

        return JobResponse(
            id=job["id"],
            kaggle_url=job["kaggle_url"],
            status=JobStatus(job["status"]),
            created_at=job["created_at"],
            updated_at=job["updated_at"],
        )

    except Exception as e:
        logger.error("create_job_failed", error=str(e), error_type=type(e).__name__)
        # Don't expose internal error details to clients
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create job. Please check your Kaggle URL and try again.",
        )


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status: str | None = Query(None, description="Filter by job status"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
) -> JobListResponse:
    """List all jobs with optional status filtering."""
    manager = get_job_manager()

    # Validate status if provided (not empty string)
    validated_status: str | None = None
    if status is not None:
        status = status.strip()
        if status:  # Non-empty after stripping
            if status not in VALID_STATUSES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status '{status}'. Valid values: {sorted(VALID_STATUSES)}",
                )
            validated_status = status

    jobs, total = await manager.list_jobs(status=validated_status, limit=limit, offset=offset)

    return JobListResponse(
        jobs=[
            JobResponse(
                id=j["id"],
                kaggle_url=j["kaggle_url"],
                status=JobStatus(j["status"]),
                created_at=j["created_at"],
                updated_at=j["updated_at"],
            )
            for j in jobs
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{job_id}", response_model=JobDetailResponse)
async def get_job(job_id: str) -> JobDetailResponse:
    """Get detailed job information."""
    manager = get_job_manager()
    job = await manager.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    status_data = await manager.get_job_status(job_id)

    return JobDetailResponse(
        id=job["id"],
        kaggle_url=job["kaggle_url"],
        status=JobStatus(job["status"]),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        dataset_name=job.get("dataset_name"),
        current_agent=status_data.get("current_agent") if status_data else None,
        iteration_count=job.get("iteration_count", 0),
        error_message=job.get("error_message"),
        progress_percentage=status_data.get("progress_percentage", 0) if status_data else 0,
        treatment_variable=job.get("treatment_variable"),
        outcome_variable=job.get("outcome_variable"),
    )


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get lightweight job status."""
    manager = get_job_manager()
    status_data = await manager.get_job_status(job_id)

    if status_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return JobStatusResponse(
        id=status_data["id"],
        status=JobStatus(status_data["status"]),
        progress_percentage=status_data["progress_percentage"],
        current_agent=status_data.get("current_agent"),
    )


@router.get("/{job_id}/stream")
async def stream_job_status(job_id: str, request: Request):
    """SSE stream for real-time job status updates.

    Streams status change events until the job reaches a terminal state.
    Includes heartbeat every 15 seconds to keep connection alive through proxies.
    """
    from src.config import get_settings
    settings = get_settings()

    if not settings.sse_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SSE streaming is disabled",
        )

    manager = get_job_manager()

    # Verify job exists
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    async def event_generator():
        last_status = None
        heartbeat_counter = 0
        heartbeat_interval = settings.sse_heartbeat_seconds

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            status_data = await manager.get_job_status(job_id)

            if status_data and status_data.get("status") != last_status:
                last_status = status_data["status"]
                yield {
                    "event": "status",
                    "data": json.dumps(status_data),
                }

                # Terminal states end the stream
                if last_status in ("completed", "failed", "cancelled"):
                    yield {"event": "done", "data": ""}
                    break

            # Send heartbeat to keep connection alive
            heartbeat_counter += 1
            if heartbeat_counter >= heartbeat_interval:
                yield {"event": "heartbeat", "data": ""}
                heartbeat_counter = 0

            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())


@router.post("/{job_id}/cancel", response_model=CancelJobResponse)
async def cancel_job(job_id: str) -> CancelJobResponse:
    """Cancel a running job.

    This endpoint gracefully cancels a running job. If the job is already
    completed, failed, or cancelled, this is a no-op.
    """
    manager = get_job_manager()

    # Check if job exists
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    result = await manager.cancel_job(job_id)

    return CancelJobResponse(
        job_id=result["job_id"],
        was_running=result["was_running"],
        cancelled=result["cancelled"],
        status=result.get("status"),
    )


@router.delete("/{job_id}", response_model=DeleteJobResponse)
async def delete_job(
    job_id: str,
    force: bool = Query(False, description="Force delete even if job is running"),
) -> DeleteJobResponse:
    """Delete a job and all its artifacts.

    This permanently deletes:
    - Job record from Firestore
    - Analysis results from Firestore
    - Agent traces from Firestore
    - Local temp files (DataFrame, notebook)

    Use force=True to cancel a running job before deletion.
    """
    manager = get_job_manager()

    # Check if job exists
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    try:
        result = await manager.delete_job(job_id, force=force)
    except ValueError as e:
        # Job is running and force=False
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )

    return DeleteJobResponse(
        job_id=result["job_id"],
        found=result["found"],
        cancelled=result["cancelled"],
        firestore_deleted=result["firestore_deleted"],
        local_artifacts_deleted=result["local_artifacts_deleted"],
    )


@router.get("/{job_id}/results", response_model=AnalysisResultsResponse)
async def get_results(job_id: str) -> AnalysisResultsResponse:
    """Get analysis results for a completed job."""
    manager = get_job_manager()

    # Check job status
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} is not completed (status: {job['status']})",
        )

    # Get results
    results = await manager.get_results(job_id)
    if results is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Results not found for job {job_id}",
        )

    # Build response
    causal_graph = None
    cg_data = results.get("causal_graph")
    if isinstance(cg_data, dict):
        causal_graph = CausalGraphResponse(
            nodes=cg_data.get("nodes", []),
            edges=cg_data.get("edges", []),
            discovery_method=cg_data.get("discovery_method", "unknown"),
            interpretation=cg_data.get("interpretation"),
        )

    # Parse treatment effects with safe fallbacks for missing fields
    treatment_effects = []
    for e in results.get("treatment_effects", []):
        if not isinstance(e, dict):
            logger.warning("invalid_treatment_effect_entry", job_id=job_id, entry_type=type(e).__name__)
            continue
        # Skip entries missing required fields
        if not all(k in e for k in ("method", "estimand", "estimate")):
            logger.warning("incomplete_treatment_effect", job_id=job_id, keys=list(e.keys()))
            continue
        treatment_effects.append(
            TreatmentEffectResponse(
                method=e.get("method", "unknown"),
                estimand=e.get("estimand", "unknown"),
                estimate=e.get("estimate", 0.0),
                std_error=e.get("std_error", 0.0),
                ci_lower=e.get("ci_lower", 0.0),
                ci_upper=e.get("ci_upper", 0.0),
                p_value=e.get("p_value"),
                assumptions_tested=e.get("assumptions_tested", []),
            )
        )

    # Parse sensitivity results with safe fallbacks
    sensitivity = []
    for s in results.get("sensitivity_results", []):
        if not isinstance(s, dict):
            logger.warning("invalid_sensitivity_entry", job_id=job_id, entry_type=type(s).__name__)
            continue
        # Skip entries missing required fields
        if not all(k in s for k in ("method", "robustness_value", "interpretation")):
            logger.warning("incomplete_sensitivity_result", job_id=job_id, keys=list(s.keys()))
            continue
        sensitivity.append(
            SensitivityResponse(
                method=s.get("method", "unknown"),
                robustness_value=s.get("robustness_value", 0.0),
                interpretation=s.get("interpretation", "No interpretation available"),
            )
        )

    # Calculate method consensus
    method_consensus = calculate_method_consensus(treatment_effects)

    # Check if sensitivity analysis shows robustness (robustness_value > 1 typically means robust)
    sensitivity_robust = (
        len(sensitivity) > 0 and
        all(s.robustness_value > 1.0 for s in sensitivity)
    )

    # Generate executive summary
    executive_summary = generate_executive_summary(
        treatment_variable=results.get("treatment_variable"),
        outcome_variable=results.get("outcome_variable"),
        effects=treatment_effects,
        consensus=method_consensus,
        sensitivity_robust=sensitivity_robust,
    )

    # Build data context
    data_context = build_data_context(results)

    return AnalysisResultsResponse(
        job_id=job_id,
        treatment_variable=results.get("treatment_variable"),
        outcome_variable=results.get("outcome_variable"),
        executive_summary=executive_summary,
        method_consensus=method_consensus,
        data_context=data_context,
        causal_graph=causal_graph,
        treatment_effects=treatment_effects,
        sensitivity_analysis=sensitivity,
        recommendations=results.get("recommendations", []),
        notebook_url=f"/jobs/{job_id}/notebook" if results.get("notebook_path") else None,
    )


@router.get("/{job_id}/notebook")
async def download_notebook(job_id: str):
    """Download the generated Jupyter notebook."""
    manager = get_job_manager()

    # Get results to find notebook path
    results = await manager.get_results(job_id)
    if results is None or not results.get("notebook_path"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notebook not found for job {job_id}",
        )

    raw_path = results["notebook_path"]

    # Validate path is a string
    if not isinstance(raw_path, str):
        logger.warning("invalid_notebook_path_type", job_id=job_id, path_type=type(raw_path).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid notebook path configuration",
        )

    notebook_path = Path(raw_path)

    # Security: Check for symlinks BEFORE resolving (prevents symlink attacks)
    if notebook_path.is_symlink():
        logger.warning(
            "notebook_symlink_detected",
            job_id=job_id,
            path=str(notebook_path),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid notebook path",
        )

    # Now resolve and validate
    resolved_path = notebook_path.resolve()

    # Security: validate resolved path is within expected directory
    try:
        resolved_path.relative_to(Path(CAUSAL_TEMP_DIR).resolve())
    except ValueError:
        logger.warning(
            "notebook_path_traversal_attempt",
            job_id=job_id,
            raw_path=raw_path,
            resolved_path=str(resolved_path),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid notebook path",
        )

    # Verify file exists and is a regular file (not a directory or device)
    if not resolved_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notebook file not found for job {job_id}",
        )

    if not resolved_path.is_file():
        logger.warning("notebook_not_regular_file", job_id=job_id, path=str(resolved_path))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid notebook path",
        )

    # Additional check: verify the file is still within the temp dir at serve time
    # (handles TOCTOU race conditions)
    try:
        temp_dir_stat = os.stat(CAUSAL_TEMP_DIR)
        notebook_parent = resolved_path.parent
        while notebook_parent != notebook_parent.parent:
            if os.stat(notebook_parent).st_ino == temp_dir_stat.st_ino:
                break
            notebook_parent = notebook_parent.parent
        else:
            raise ValueError("Path not within temp directory")
    except (OSError, ValueError) as e:
        logger.warning("notebook_path_validation_failed", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid notebook path",
        )

    return FileResponse(
        path=str(resolved_path),
        filename=f"causal_analysis_{job_id}.ipynb",
        media_type="application/x-ipynb+json",
    )


@router.get("/{job_id}/traces", response_model=AgentTracesResponse)
async def get_traces(job_id: str) -> AgentTracesResponse:
    """Get agent reasoning traces for a job."""
    manager = get_job_manager()

    # Check job exists
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    traces = await manager.get_traces(job_id)

    return AgentTracesResponse(
        job_id=job_id,
        traces=[
            {
                "agent_name": t["agent_name"],
                "timestamp": t["timestamp"],
                "action": t["action"],
                "reasoning": t["reasoning"],
                "duration_ms": t.get("duration_ms", 0),
            }
            for t in traces
        ],
    )
