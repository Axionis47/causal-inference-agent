import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.analysis import prompts, service
from backend.analysis.contracts import (
    AgentDetail,
    AnalysisSummary,
    AnalysisView,
    ApproveAnalysis,
    PrepareAnalysis,
    RejectAnalysis,
    SharedAttemptDetail,
)
from backend.analysis.readiness import AnalysisReadiness
from backend.auth import Db, Principal
from backend.db import engine
from backend.handoffs import ConversationScopeReader
from backend.models import AnalysisRun
from backend.storage import files


def router(conversation_reader: ConversationScopeReader) -> APIRouter:
    router = APIRouter(tags=["analysis"])


    @router.get("/api/conversations/{conversation_id}/analysis-readiness", response_model=AnalysisReadiness)
    def readiness(conversation_id: UUID, db: Db, user: Principal) -> AnalysisReadiness:
        return service.readiness(db, conversation_reader.read(db, user.id, conversation_id))


    @router.get("/api/conversations/{conversation_id}/analyses", response_model=list[AnalysisSummary])
    def list_analyses(conversation_id: UUID, db: Db, user: Principal) -> list[AnalysisSummary]:
        conversation_reader.read(db, user.id, conversation_id)
        return [
            AnalysisSummary.model_validate(
                {
                    **{key: getattr(row, key) for key in ("id", "question", "status", "stage", "created_at")},
                    "read_only": row.inputs.get("contract") != prompts.VERSION,
                }
            )
            for row in db.scalars(
                select(AnalysisRun)
                .where(AnalysisRun.conversation_id == conversation_id)
                .order_by(AnalysisRun.created_at.desc())
            )
        ]


    @router.post("/api/conversations/{conversation_id}/analyses", response_model=AnalysisView, status_code=202)
    def prepare(conversation_id: UUID, body: PrepareAnalysis, db: Db, user: Principal) -> AnalysisView:
        return service.prepare(db, conversation_reader.read(db, user.id, conversation_id, lock=True), user.id, body)


    @router.get("/api/analyses/{run_id}", response_model=AnalysisView)
    def read(run_id: UUID, db: Db, user: Principal) -> AnalysisView:
        return service.view(db, service.owned(db, user.id, run_id))

    @router.get("/api/analyses/{run_id}/agents/{agent_id}", response_model=AgentDetail)
    def agent_detail(run_id: UUID, agent_id: UUID, db: Db, user: Principal) -> dict[str, Any]:
        from backend.analysis.agent_views import detail
        return detail(db, service.owned(db, user.id, run_id), agent_id)


    @router.post("/api/analyses/{run_id}/approve", response_model=AnalysisView, status_code=202)
    def approve(run_id: UUID, body: ApproveAnalysis, db: Db, user: Principal) -> AnalysisView:
        return service.approve(db, service.owned(db, user.id, run_id, lock=True), body)


    @router.post("/api/analyses/{run_id}/retry", response_model=AnalysisView, status_code=202)
    def retry(run_id: UUID, db: Db, user: Principal) -> AnalysisView:
        return service.retry(db, service.owned(db, user.id, run_id, lock=True))


    @router.post("/api/analyses/{run_id}/reject", response_model=AnalysisView)
    def reject(run_id: UUID, body: RejectAnalysis, db: Db, user: Principal) -> AnalysisView:
        return service.reject(db, service.owned(db, user.id, run_id, lock=True), body)

    @router.post("/api/analyses/{run_id}/revise-program", response_model=AnalysisView, status_code=202)
    def revise_program(run_id: UUID, body: RejectAnalysis, db: Db, user: Principal) -> AnalysisView:
        return service.reject(db, service.owned(db, user.id, run_id, lock=True), body, request_revision=True)


    @router.get("/api/analyses/{run_id}/events")
    def stream(
        run_id: UUID,
        request: Request,
        db: Db,
        user: Principal,
        after: int = Query(default=0, ge=0),
        last_event_id: str | None = Header(default=None),
    ) -> StreamingResponse:
        service.owned(db, user.id, run_id)
        # Authorize once, then release the request's pooled connection before the
        # long-lived stream. Each replay batch owns its own short transaction.
        db.rollback()
        try:
            cursor = max(after, int(last_event_id or 0))
        except ValueError as error:
            raise HTTPException(400, "Invalid event cursor.") from error

        def batch(position: int) -> tuple[list[str], int, str]:
            with Session(engine()) as session:
                run = service.owned(session, user.id, run_id)
                events = service.events(session, run_id, position)
                return (
                    [f"id: {e.sequence}\nevent: checkpoint\ndata: {e.model_dump_json()}\n\n" for e in events],
                    events[-1].sequence if events else position,
                    run.status,
                )

        async def generate() -> AsyncIterator[str]:
            position = cursor
            for _ in range(240):
                if await request.is_disconnected():
                    return
                rows, position, status = await asyncio.to_thread(batch, position)
                for row in rows:
                    yield row
                if status in {"awaiting_approval", "blocked", "succeeded", "failed"}:
                    yield "event: settled\ndata: " + json.dumps({"status": status}) + "\n\n"
                    return
                yield ": keepalive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )


    @router.get("/api/analyses/{run_id}/files/{filename}")
    def artifact(run_id: UUID, filename: str, db: Db, user: Principal) -> Response:
        run = service.owned(db, user.id, run_id)
        if filename == "context.json" and run.context_revision_id:
            from backend.context.service import read_context

            raw = read_context(db, run.version_id, run.context_revision_id, []).model_dump_json(indent=2).encode()
        elif filename == "snapshot.json" and run.snapshot:
            raw = json.dumps(run.snapshot, indent=2).encode()
        elif filename == "trajectory.json":
            raw = service.view(db, run).model_dump_json(indent=2).encode()
        elif filename == "graph.json":
            frame = service.view(db, run).frame
            if frame is None:
                raise HTTPException(404, "Analysis graph not available.")
            raw = frame.model_dump_json(indent=2).encode()
        elif filename.startswith(("review-rejection-", "checkpoint-")) or filename == "frozen-context.json":
            from backend.analysis import artifacts

            try:
                raw = artifacts.read(run, filename)
            except ValueError as error:
                raise HTTPException(404, "Analysis artifact not found.") from error
        else:
            from backend.analysis.records import programs

            saved = programs(run)
            current = saved[-1] if saved else None
            code_match = re.fullmatch(r"program-(\d+)\.py", filename)
            diagnostic_match = re.fullmatch(r"diagnostic-(\d+)\.py", filename)
            output_match = re.fullmatch(r"execution-(\d+)-(.+)", filename)
            if filename == "code.py" and current:
                raw = current["program"]["code"].encode()
            elif code_match and int(code_match[1]) < len(saved):
                raw = saved[int(code_match[1])]["program"]["code"].encode()
            elif diagnostic_match and f"diagnostic:{int(diagnostic_match[1])}" in run.receipts:
                record = run.receipts[f"diagnostic:{int(diagnostic_match[1])}"]
                assert isinstance(record, dict)
                raw = str(record["code"]).encode()
            elif filename == "code.py" and run.snapshot and "program" in run.snapshot:
                program = run.snapshot["program"]
                assert isinstance(program, dict)
                raw = str(program["code"]).encode()
            else:
                execution_key = (
                    f"execution:{int(output_match[1])}"
                    if output_match
                    else f"execution:{current['revision']}"
                    if current
                    else "execution"
                )
                execution = run.receipts.get(execution_key)
                name = output_match[2] if output_match else filename
                record = execution.get("files", {}).get(name) if isinstance(execution, dict) else None
                if not isinstance(record, dict):
                    raise HTTPException(404, "Analysis artifact not found.")
                raw = files().read(str(record["key"]), str(record["sha256"]))
        image = filename.endswith("figure.png")
        return Response(
            raw,
            media_type="image/png" if image else "application/octet-stream",
            headers={
                "Content-Disposition": f"{'inline' if image else 'attachment'}; filename*=UTF-8''{quote(filename, safe='')}"
            },
        )


    @router.get("/api/analyses/{run_id}/checkpoints/{receipt_id}/input")
    def checkpoint_input(run_id: UUID, receipt_id: str, db: Db, user: Principal) -> dict[str, object]:
        run = service.owned(db, user.id, run_id)
        for key, value in run.receipts.items():
            if key.startswith("checkpoint:") and isinstance(value, dict) and value.get("id") == receipt_id:
                record = run.receipts.get("input:" + key.removeprefix("checkpoint:"))
                if isinstance(record, dict):
                    return {"receipt_id": receipt_id, "input_hash": record["hash"], "packet": record["packet"]}
        from backend.analysis.tasks import input_for_receipt
        packet = input_for_receipt(db, run_id, receipt_id)
        if packet is not None:
            from backend.context.service import fingerprint
            return {"receipt_id": receipt_id, "input_hash": fingerprint(packet), "packet": packet}
        raise HTTPException(404, "Checkpoint input not found.")


    @router.get("/api/analyses/{run_id}/provider-attempts/{attempt_id}", response_model=SharedAttemptDetail)
    def provider_attempt(run_id: UUID, attempt_id: UUID, db: Db, user: Principal) -> dict[str, Any]:
        from backend.analysis.tasks import shared_attempt

        service.owned(db, user.id, run_id)
        record = shared_attempt(db, run_id, attempt_id)
        if record is None:
            raise HTTPException(404, "Provider attempt not found.")
        return record

    return router
