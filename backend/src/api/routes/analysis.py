"""Analysis-run routes: the tile view and artifact bytes.

GET /jobs/{job_id}/analysis reconstructs the full per-agent view from the
persisted run state (this is also the reopen path for old jobs). Artifact
bytes are served from the job's analysis dir through the traversal-safe
resolver only.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.analysis_v2.core import SPINE, AnalysisRunState, stage_index
from src.analysis_v2.persistence import load_run, read_artifact_bytes
from src.api.rate_limit import limiter

router = APIRouter(prefix="/jobs", tags=["analysis"])


class PlanDecisionRequest(BaseModel):
    decision: Literal["confirm", "reject"]
    edits: dict[str, str] | None = None
    reason: str | None = Field(default=None, max_length=500)


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
    plan_gate = None
    if run.plan_critique is not None:
        card = run.plan_critique.confirmation_card
        plan_gate = {
            "status": run.plan_critique.status.value,
            "reasons": run.plan_critique.reasons,
            "missing_required": run.plan_critique.missing_required,
            "summary": run.plan_critique.summary,
            "confirmation_card": card.model_dump(mode="json") if card else None,
        }
    return {
        "job_id": run.job_id,
        "status": run.status.value,
        "current_state": run.current_state.value,
        "stage_index": stage_index(run.current_state),
        "total_stages": len(SPINE),
        "causal_question": run.causal_question,
        "error_message": run.error_message,
        "spec_summary": spec_summary,
        "plan_gate": plan_gate,
        "method_plan": (
            run.method_plan.model_dump(mode="json") if run.method_plan else None
        ),
        "selected_design": (
            run.selected_design.model_dump(mode="json") if run.selected_design else None
        ),
        "notebook": (
            {
                "status": run.notebook_verification.notebook_status.value,
                "attempts": run.notebook_verification.attempts,
                "verified_artifact_id": run.notebook_verification.verified_notebook_artifact_id,
                "html_artifact_id": run.notebook_verification.html_preview_artifact_id,
            }
            if run.notebook_verification is not None
            else None
        ),
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


@router.post("/{job_id}/plan")
@limiter.limit("30/minute")
async def submit_plan_decision(
    request: Request, job_id: str, body: PlanDecisionRequest
) -> dict:
    from src.analysis_v2.runner.resume import (
        InvalidPlanEdits,
        NoAnalysisRun,
        NotWaitingForUser,
        apply_plan_decision,
    )
    from src.jobs import get_job_manager

    if body.decision == "reject" and not (body.reason or "").strip():
        raise HTTPException(status_code=422, detail="rejecting the plan requires a reason")
    try:
        return await apply_plan_decision(
            job_id,
            get_job_manager(),
            decision=body.decision,
            edits=body.edits,
            reason=body.reason,
        )
    except NoAnalysisRun as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except NotWaitingForUser as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except InvalidPlanEdits as exc:
        raise HTTPException(status_code=422, detail=str(exc))


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
