"""Job management API routes."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from src.agents.base import JobStatus
from src.api.schemas import (
    AgentTracesResponse,
    AnalysisResultsResponse,
    CausalGraphResponse,
    CreateJobRequest,
    JobDetailResponse,
    JobListResponse,
    JobResponse,
    JobStatusResponse,
    SensitivityResponse,
    TreatmentEffectResponse,
)
from src.jobs.manager import get_job_manager
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(request: CreateJobRequest) -> JobResponse:
    """Create a new analysis job.

    Submit a Kaggle dataset URL to start causal inference analysis.
    """
    manager = get_job_manager()

    try:
        job_id = await manager.create_job(
            kaggle_url=request.kaggle_url,
            treatment_variable=request.treatment_variable,
            outcome_variable=request.outcome_variable,
            preferences=request.analysis_preferences,
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
        logger.error("create_job_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(e)}",
        )


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> JobListResponse:
    """List all jobs with optional status filtering."""
    manager = get_job_manager()

    jobs, total = await manager.list_jobs(status=status, limit=limit, offset=offset)

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


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job(job_id: str) -> None:
    """Cancel a running job."""
    manager = get_job_manager()

    # Check if job exists
    job = await manager.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    # Try to cancel
    await manager.cancel_job(job_id)


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
    if results.get("causal_graph"):
        cg = results["causal_graph"]
        causal_graph = CausalGraphResponse(
            nodes=cg["nodes"],
            edges=cg["edges"],
            discovery_method=cg["discovery_method"],
        )

    treatment_effects = [
        TreatmentEffectResponse(
            method=e["method"],
            estimand=e["estimand"],
            estimate=e["estimate"],
            std_error=e["std_error"],
            ci_lower=e["ci_lower"],
            ci_upper=e["ci_upper"],
            p_value=e.get("p_value"),
            assumptions_tested=e.get("assumptions_tested", []),
        )
        for e in results.get("treatment_effects", [])
    ]

    sensitivity = [
        SensitivityResponse(
            method=s["method"],
            robustness_value=s["robustness_value"],
            interpretation=s["interpretation"],
        )
        for s in results.get("sensitivity_results", [])
    ]

    return AnalysisResultsResponse(
        job_id=job_id,
        treatment_variable=results.get("treatment_variable"),
        outcome_variable=results.get("outcome_variable"),
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

    notebook_path = results["notebook_path"]

    return FileResponse(
        path=notebook_path,
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
