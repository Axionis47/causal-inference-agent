"""Authenticated chat HTTP boundary."""

from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend.auth import Db, Principal
from backend.chat import service
from backend.chat.contracts import (
    ConversationDetail,
    ConversationView,
    CreateConversation,
    SendTurn,
    TurnView,
)
from backend.handoffs import AnalysisDiscussionReader
from backend.storage import files


def router(analysis_reader: AnalysisDiscussionReader) -> APIRouter:
    router = APIRouter(tags=["chat"])


    @router.get("/api/versions/{version_id}/conversations", response_model=list[ConversationView])
    def conversations(version_id: UUID, db: Db, user: Principal) -> list[ConversationView]:
        return service.list_conversations(db, user.id, version_id)


    @router.post("/api/versions/{version_id}/conversations", response_model=ConversationView, status_code=201)
    def create(version_id: UUID, body: CreateConversation, db: Db, user: Principal) -> ConversationView:
        return service.create(db, user.id, version_id, body)


    @router.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
    def conversation(conversation_id: UUID, db: Db, user: Principal) -> ConversationDetail:
        return service.detail(db, service.owned(db, user.id, conversation_id))


    @router.post("/api/conversations/{conversation_id}/turns", response_model=TurnView, status_code=202)
    def send(conversation_id: UUID, body: SendTurn, db: Db, user: Principal) -> TurnView:
        return service.send(db, service.owned(db, user.id, conversation_id, lock=True), body, analysis_reader)


    @router.post(
        "/api/conversations/{conversation_id}/turns/{turn_id}/retry", response_model=TurnView, status_code=202
    )
    def retry(conversation_id: UUID, turn_id: UUID, db: Db, user: Principal) -> TurnView:
        return service.retry(db, service.owned(db, user.id, conversation_id, lock=True), turn_id)


    @router.post(
        "/api/conversations/{conversation_id}/turns/{turn_id}/apply-context", response_model=ConversationDetail
    )
    def apply_context(conversation_id: UUID, turn_id: UUID, db: Db, user: Principal) -> ConversationDetail:
        conversation = service.owned(db, user.id, conversation_id, lock=True)
        service.apply_context(db, conversation, user.id, turn_id)
        return service.detail(db, conversation)


    @router.get("/api/conversations/{conversation_id}/artifacts/{artifact_id}/files/{filename}")
    def artifact_file(
        conversation_id: UUID, artifact_id: UUID, filename: str, db: Db, user: Principal
    ) -> Response:
        service.owned(db, user.id, conversation_id)
        artifact = service.artifact_record(db, conversation_id, artifact_id)
        record = artifact.files.get(filename)
        if not isinstance(record, dict):
            raise HTTPException(404, "Artifact file not found.")
        raw = files().read(str(record["key"]), str(record["sha256"]))
        image = filename == "figure.png"
        return Response(
            raw,
            media_type="image/png" if image else "application/octet-stream",
            headers={
                "Content-Disposition": f"{'inline' if image else 'attachment'}; filename*=UTF-8''{quote(filename, safe='')}"
            },
        )

    return router
