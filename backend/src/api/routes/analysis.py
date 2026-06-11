"""Analysis-run routes: the tile view and artifact bytes.

GET /jobs/{job_id}/analysis reconstructs the full per-agent view from the
persisted run state (this is also the reopen path for old jobs). Artifact
bytes are served from the job's analysis dir through the traversal-safe
resolver only.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from src.analysis_v2.core import AnalysisRunState, stage_index
from src.analysis_v2.persistence import load_run, read_artifact_bytes
from src.api.rate_limit import limiter

router = APIRouter(prefix="/jobs", tags=["analysis"])


def _view(run: AnalysisRunState) -> dict:
    spec = run.causal_spec
    spec_summary = None
    if spec is not None:
        spec_summary = {
            "question_type": spec.question_type.value,
            "confidence": spec.confidence.value,
            "outcome": spec.outcome.column,
            "treatment": spec.treatment.column,
        }
    return {
        "job_id": run.job_id,
        "status": run.status.value,
        "current_state": run.current_state.value,
        "stage_index": stage_index(run.current_state),
        "total_stages": 13,
        "causal_question": run.causal_question,
        "error_message": run.error_message,
        "spec_summary": spec_summary,
        "agents": [
            {
                "agent": r.agent,
                "stage": r.stage.value,
                "status": r.status.value,
                "public_summary": r.public_summary,
                "current_step": r.current_step,
                "warnings": r.warnings,
                "artifact_ids": r.artifact_ids,
                "tool_call_count": len(r.tool_calls),
                "tokens": r.tokens.model_dump(),
                "cost_usd": r.cost_usd,
                "elapsed_seconds": r.elapsed_seconds,
                "attempt": r.attempt,
            }
            for r in run.agent_runs
        ],
        "artifacts": [
            {
                "artifact_id": a.artifact_id,
                "kind": a.kind.value,
                "stage": a.stage.value,
                "agent": a.agent,
                "title": a.title,
                "summary": a.summary,
                "media_type": a.media_type,
                "created_at": a.created_at.isoformat(),
            }
            for a in run.artifact_registry.artifacts
        ],
        "costs": {
            "total_input_tokens": run.total_tokens.input_tokens,
            "total_output_tokens": run.total_tokens.output_tokens,
            "total_cost_usd": run.total_cost_usd,
            "total_tool_calls": sum(len(r.tool_calls) for r in run.agent_runs),
        },
        "events": [
            {
                "sequence": e.sequence,
                "from_state": e.from_state.value,
                "to_state": e.to_state.value,
                "agent_name": e.agent_name,
                "timestamp": e.timestamp.isoformat(),
                "warnings": e.warnings,
            }
            for e in run.state_events
        ],
    }


@router.get("/{job_id}/analysis")
@limiter.limit("120/minute")
async def get_analysis_view(request: Request, job_id: str) -> dict:
    run = await load_run(job_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} has no analysis run")
    return _view(run)


@router.get("/{job_id}/analysis/artifacts/{artifact_id:path}")
@limiter.limit("240/minute")
async def get_analysis_artifact(request: Request, job_id: str, artifact_id: str) -> Response:
    run = await load_run(job_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} has no analysis run")
    artifact = run.artifact_registry.get(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"unknown artifact {artifact_id!r}")
    try:
        payload = read_artifact_bytes(job_id, artifact.path)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail="artifact file is missing")
    return Response(content=payload, media_type=artifact.media_type)
