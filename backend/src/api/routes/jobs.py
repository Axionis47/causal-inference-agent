"""Job management API routes."""

import asyncio
import io
import json
import os
import zipfile
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from src.analysis_v2.state import JobStatus
from src.api.idempotency import get_idempotency_store
from src.api.rate_limit import limiter
from src.api.schemas import (
    AgentTracesResponse,
    AnalysisResultsResponse,
    ApprovalRequest,
    ApprovalResultResponse,
    ApprovalSnapshotResponse,
    CancelJobResponse,
    CausalGraphResponse,
    ConfirmDatasetResponse,
    CreateJobRequest,
    DatasetInputsResponse,
    DatasetRowsPage,
    DatasetViewResponse,
    DeleteJobResponse,
    JobDetailResponse,
    JobListResponse,
    JobResponse,
    JobStatusResponse,
    SensitivityResponse,
    TreatmentEffectResponse,
    UpdateInputsRequest,
)
from src.api.utils import (
    build_data_context,
    build_dataset_view_from_persisted,
    build_dataset_view_from_state,
    calculate_method_consensus,
    generate_executive_summary,
    generate_narrative_summary,
)
from src.jobs.manager import get_job_manager
from src.logging_config.structured import get_logger
from src.storage.cleanup import CAUSAL_TEMP_DIR
from src.storage.job_data import read_file_page

logger = get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Upper bound on rows returned per dataset page, so a large file can never be
# pulled into one response.
_MAX_ROWS_PER_PAGE = 500

# Valid job statuses for query parameter validation
VALID_STATUSES = {s.value for s in JobStatus}


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def create_job(
    request: Request,
    body: CreateJobRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> JobResponse:
    """Create a new analysis job.

    Submit a Kaggle dataset URL to start causal inference analysis.

    Supply Idempotency-Key for safe retries. A repeat request with the
    same key (within 24h, scoped to your API key) returns the original
    job instead of starting a duplicate analysis.
    """
    # slowapi requires 'request' as first param with type starlette.requests.Request
    # FastAPI will inject the Pydantic body via the 'body' parameter
    manager = get_job_manager()
    store = get_idempotency_store()

    # Idempotent retry: return the original job if we've seen this key
    if idempotency_key:
        existing_job_id = store.lookup(x_api_key, idempotency_key)
        if existing_job_id is not None:
            existing = await manager.get_job(existing_job_id)
            if existing is not None:
                logger.info(
                    "idempotent_retry_hit",
                    idempotency_key=idempotency_key[:16],
                    job_id=existing_job_id,
                )
                return JobResponse(
                    id=existing["id"],
                    kaggle_url=existing["kaggle_url"],
                    status=JobStatus(existing["status"]),
                    created_at=existing["created_at"],
                    updated_at=existing["updated_at"],
                )
            # Cached job_id no longer resolves (deleted, expired). Fall
            # through and create a new job; remember will overwrite.

    try:
        job_id = await manager.create_job(
            kaggle_url=body.kaggle_url,
            causal_question=body.causal_question,
            orchestrator_mode=body.orchestrator_mode,
            user_context=body.user_context,
        )

        # Get the created job
        job = await manager.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create job",
            )

        # Remember (api_key_hash, idempotency_key) -> job_id for future retries
        if idempotency_key:
            store.remember(x_api_key, idempotency_key, job_id)

        return JobResponse(
            id=job["id"],
            kaggle_url=job["kaggle_url"],
            status=JobStatus(job["status"]),
            created_at=job["created_at"],
            updated_at=job["updated_at"],
        )

    except RuntimeError as e:
        # Backpressure: server at capacity
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )
    except Exception as e:
        logger.error("create_job_failed", error=str(e), error_type=type(e).__name__)
        # Don't expose internal error details to clients
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create job. Please check your Kaggle URL and try again.",
        )


@router.get("", response_model=JobListResponse)
@limiter.limit("30/minute")
async def list_jobs(
    request: Request,
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
                    status_code=400,
                    detail=f"Invalid status '{status}'. Valid values: {sorted(VALID_STATUSES)}",
                )
            validated_status = status

    jobs, total = await manager.list_jobs(status=validated_status, limit=limit, offset=offset)

    # Tolerate legacy records: a job persisted before the analysis-engine
    # teardown can carry a status the trimmed JobStatus enum no longer knows
    # (e.g. "completed"). Skip those rather than 500 the whole list, and adjust
    # the count so pagination stays honest.
    items: list[JobResponse] = []
    skipped = 0
    for j in jobs:
        if j.get("status") not in VALID_STATUSES:
            skipped += 1
            continue
        items.append(
            JobResponse(
                id=j["id"],
                kaggle_url=j["kaggle_url"],
                status=JobStatus(j["status"]),
                created_at=j["created_at"],
                updated_at=j["updated_at"],
                dataset_name=j.get("dataset_name"),
                causal_question=j.get("causal_question"),
                treatment_variable=j.get("treatment_variable"),
                outcome_variable=j.get("outcome_variable"),
                iteration_count=j.get("iteration_count", 0),
            )
        )
    if skipped:
        logger.info("jobs_list_skipped_legacy_status", skipped=skipped)

    return JobListResponse(
        jobs=items,
        total=max(0, total - skipped),
        limit=limit,
        offset=offset,
    )


@router.get("/{job_id}", response_model=JobDetailResponse)
@limiter.limit("60/minute")
async def get_job(request: Request, job_id: str) -> JobDetailResponse:
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
        causal_question=job.get("causal_question"),
        treatment_variable=job.get("treatment_variable"),
        outcome_variable=job.get("outcome_variable"),
    )


@router.get("/{job_id}/dataset", response_model=DatasetViewResponse)
@limiter.limit("60/minute")
async def get_dataset_view(request: Request, job_id: str) -> DatasetViewResponse:
    """Return the live + persisted view of the job's dataset.

    Backs the frontend Data panel: download status, Kaggle metadata,
    and computed profile, each as an independent block. Reads live
    in-memory state when the job is active and falls back to the
    persisted record for completed / evicted jobs.
    """
    manager = get_job_manager()
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    state = manager.get_active_state(job_id)
    if state is None and job.get("status") in (
        JobStatus.AWAITING_APPROVAL.value,
        JobStatus.CONFIRMED.value,
    ):
        # A job parked at the data gate (or confirmed and stopped) has no active
        # state, but its full state (the data being reviewed, or the confirmed
        # record) lives in the parked store.
        state = await manager.get_parked_state(job_id)
    if state is not None:
        return build_dataset_view_from_state(state)

    results = await manager.get_results(job_id)
    return build_dataset_view_from_persisted(job, results)


@router.get(
    "/{job_id}/dataset/files/{file_name}/rows", response_model=DatasetRowsPage
)
@limiter.limit("120/minute")
async def get_dataset_rows(
    request: Request,
    job_id: str,
    file_name: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=_MAX_ROWS_PER_PAGE),
) -> DatasetRowsPage:
    """Return one page of raw rows for a downloaded file.

    Rows are read on demand from the durable per-job bundle, resolved via the
    manifest, so the Data panel can page through datasets of any size without
    the rows ever being embedded in the dataset-view response. 404 if the job,
    its manifest, or the named file is absent.
    """
    manager = get_job_manager()
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    page = read_file_page(job_id, file_name, offset, limit)
    if page is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No readable file {file_name!r} for job {job_id}",
        )
    return DatasetRowsPage(**page)


@router.patch("/{job_id}/inputs", response_model=DatasetInputsResponse)
@limiter.limit("30/minute")
async def update_dataset_inputs(
    request: Request, job_id: str, body: UpdateInputsRequest
) -> DatasetInputsResponse:
    """Apply analyst edits to a job at the data-review gate.

    The analyst may refine the causal question and set or clear the time column;
    a time column that is not a real column is rejected with 422. Valid only
    while the job is awaiting data review.
    """
    manager = get_job_manager()
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    if job.get("status") != JobStatus.AWAITING_APPROVAL.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Job {job_id} is not awaiting data review "
                f"(current status: {job.get('status')})"
            ),
        )
    try:
        result = await manager.set_dataset_inputs(
            job_id,
            causal_question=body.causal_question,
            time_column=body.time_column,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return DatasetInputsResponse(**result)


@router.post("/{job_id}/confirm", response_model=ConfirmDatasetResponse)
@limiter.limit("30/minute")
async def confirm_dataset(request: Request, job_id: str) -> ConfirmDatasetResponse:
    """Confirm the dataset at the data-review gate.

    Stores the data + inputs as the confirmed-dataset record and ends the input
    flow. The analysis is NOT started; launch it later via POST /jobs/{id}/run.
    Valid only while the job is awaiting data review.
    """
    manager = get_job_manager()
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    if job.get("status") != JobStatus.AWAITING_APPROVAL.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Job {job_id} is not awaiting data review "
                f"(current status: {job.get('status')})"
            ),
        )
    try:
        result = await manager.confirm_dataset(job_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return ConfirmDatasetResponse(**result)


@router.post("/{job_id}/run", response_model=ApprovalResultResponse)
@limiter.limit("30/minute")
async def run_analysis(request: Request, job_id: str) -> ApprovalResultResponse:
    """Launch the analysis pipeline on a CONFIRMED dataset.

    The dataset + inputs must already be confirmed (POST /confirm). Respawns the
    orchestrator past the data gate. A FAILED job retries from the same
    confirmed dataset; jobs that never passed the data gate are refused.
    """
    manager = get_job_manager()
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    if job.get("status") not in (JobStatus.CONFIRMED.value, JobStatus.FAILED.value):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Job {job_id} has no confirmed dataset ready to run "
                f"(current status: {job.get('status')})"
            ),
        )
    try:
        result = await manager.run_analysis(job_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return ApprovalResultResponse(
        job_id=job_id, resumed=result["resumed"], status=result["status"]
    )


@router.get("/{job_id}/status", response_model=JobStatusResponse)
@limiter.limit("120/minute")
async def get_job_status(request: Request, job_id: str) -> JobStatusResponse:
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
@limiter.limit("5/minute")
async def stream_job_status(request: Request, job_id: str):
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

    # SSE TTL: max duration to prevent zombie connections (1 hour)
    max_sse_seconds = 3600

    async def event_generator():
        import time

        last_status = None
        last_seen_event_index = 0
        heartbeat_counter = 0
        heartbeat_interval = settings.sse_heartbeat_seconds
        start_time = time.monotonic()

        try:
            while True:
                # TTL guard: close if stream exceeds max duration
                if time.monotonic() - start_time > max_sse_seconds:
                    yield {"event": "timeout", "data": json.dumps({"reason": "SSE TTL exceeded"})}
                    break

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

                # Emit new agent-level SSE events
                new_events = manager.get_sse_events(job_id, after_index=last_seen_event_index)
                for event in new_events:
                    yield {
                        "event": "agent_event",
                        "data": json.dumps(event),
                    }
                last_seen_event_index += len(new_events)

                # Send heartbeat to keep connection alive
                heartbeat_counter += 1
                if heartbeat_counter >= heartbeat_interval:
                    yield {"event": "heartbeat", "data": ""}
                    heartbeat_counter = 0

                await asyncio.sleep(1)
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            logger.debug("sse_stream_closed", job_id=job_id)

    return EventSourceResponse(event_generator())


@router.get("/{job_id}/approval", response_model=ApprovalSnapshotResponse)
@limiter.limit("60/minute")
async def get_approval_snapshot(
    request: Request, job_id: str
) -> ApprovalSnapshotResponse:
    """Return the data/DAG/flags snapshot for a job parked at the human-approval gate.

    Returns 404 if the job has no parked state (either it never reached
    the gate or it has already been approved/rejected).
    """
    manager = get_job_manager()
    snapshot = await manager.get_parked_snapshot(job_id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} is not parked at the approval gate",
        )
    return ApprovalSnapshotResponse(**snapshot)


@router.post("/{job_id}/approval", response_model=ApprovalResultResponse)
@limiter.limit("30/minute")
async def submit_approval(
    request: Request, job_id: str, body: ApprovalRequest
) -> ApprovalResultResponse:
    """Apply a human approve/reject decision to a job parked at the data gate.

    APPROVED: delegates to the confirm path; the dataset becomes the
    CONFIRMED record and no pipeline is started.

    REJECTED: job is marked FAILED with `reason` as the error_message and
    the parked state is cleared. `reason` is required when
    decision == "rejected".
    """
    from datetime import datetime, timezone

    from src.domain.approval import (
        ApprovalDecision,
        DagEdit,
        HumanApproval,
    )

    job = await get_job_manager().get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    if job.get("status") != JobStatus.AWAITING_APPROVAL.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Job {job_id} is not awaiting approval "
                f"(current status: {job.get('status')})"
            ),
        )

    try:
        approval = HumanApproval(
            decision=ApprovalDecision(body.decision),
            granted_at=datetime.now(timezone.utc),
            granted_by=body.granted_by,
            dag_edits=(
                DagEdit(**body.dag_edits.model_dump(exclude_none=True))
                if body.dag_edits is not None
                else None
            ),
            appended_context=body.appended_context,
            reason=body.reason,
        )
    except ValueError as exc:
        # E.g. rejected-without-reason.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    from src.jobs.gate_resume import RevisionLimitReached

    try:
        result = await get_job_manager().resume_from_approval(job_id, approval)
    except RevisionLimitReached as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return ApprovalResultResponse(
        job_id=job_id, resumed=result["resumed"], status=result["status"]
    )


@router.post("/{job_id}/cancel", response_model=CancelJobResponse)
@limiter.limit("30/minute")
async def cancel_job(request: Request, job_id: str) -> CancelJobResponse:
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
@limiter.limit("10/minute")
async def delete_job(
    request: Request,
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
@limiter.limit("30/minute")
async def get_results(request: Request, job_id: str) -> AnalysisResultsResponse:
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

    # Generate narrative summary
    narrative_summary = generate_narrative_summary(
        treatment_effects=treatment_effects,
        sensitivity_results=sensitivity,
        treatment_var=results.get("treatment_variable"),
        outcome_var=results.get("outcome_variable"),
    )

    return AnalysisResultsResponse(
        job_id=job_id,
        treatment_variable=results.get("treatment_variable"),
        outcome_variable=results.get("outcome_variable"),
        executive_summary=executive_summary,
        method_consensus=method_consensus,
        data_context=data_context,
        narrative_summary=narrative_summary,
        causal_graph=causal_graph,
        treatment_effects=treatment_effects,
        sensitivity_analysis=sensitivity,
        recommendations=results.get("recommendations", []),
        notebook_url=f"/jobs/{job_id}/notebook" if results.get("notebook_path") else None,
        decision_log=results.get("decisions", []),
    )


@router.get("/{job_id}/notebook")
@limiter.limit("30/minute")
async def download_notebook(request: Request, job_id: str):
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


@router.get("/{job_id}/notebook/bundle")
@limiter.limit("30/minute")
async def download_notebook_bundle(request: Request, job_id: str):
    """Download notebook + data as a reproducible zip bundle."""
    manager = get_job_manager()

    results = await manager.get_results(job_id)
    if results is None or not results.get("notebook_path"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notebook not found for job {job_id}",
        )

    notebook_path = Path(results["notebook_path"])
    if not notebook_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notebook file not found for job {job_id}",
        )

    # Create in-memory zip with notebook + data file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add notebook
        zf.write(str(notebook_path), f"causal_analysis_{job_id}.ipynb")

        # Find and add co-located data file
        notebook_dir = notebook_path.parent
        for ext in (".parquet", ".csv"):
            data_file = notebook_dir / f"data_{job_id}{ext}"
            if data_file.exists():
                zf.write(str(data_file), data_file.name)
                break

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="causal_analysis_{job_id}.zip"'
        },
    )


@router.get("/{job_id}/traces", response_model=AgentTracesResponse)
@limiter.limit("30/minute")
async def get_traces(
    request: Request,
    job_id: str,
    agent_name: str | None = Query(None, description="Filter by agent name"),
    limit: int = Query(200, ge=1, le=500, description="Max traces to return"),
) -> AgentTracesResponse:
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

    # Optional agent filtering
    if agent_name:
        traces = [t for t in traces if t.get("agent_name") == agent_name]

    # Apply limit
    traces = traces[:limit]

    # Build response with all fields
    trace_responses = []
    total_duration_ms = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for t in traces:
        duration = t.get("duration_ms", 0)
        token_usage = t.get("token_usage", {})
        total_duration_ms += duration
        total_input_tokens += token_usage.get("input_tokens", 0)
        total_output_tokens += token_usage.get("output_tokens", 0)

        trace_responses.append({
            "agent_name": t["agent_name"],
            "timestamp": t["timestamp"],
            "action": t["action"],
            "reasoning": t.get("reasoning", ""),
            "duration_ms": duration,
            "inputs": t.get("inputs", {}),
            "outputs": t.get("outputs", {}),
            "tools_called": t.get("tools_called", []),
            "token_usage": token_usage,
        })

    return AgentTracesResponse(
        job_id=job_id,
        traces=trace_responses,
        total_traces=len(trace_responses),
        total_duration_ms=total_duration_ms,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )
