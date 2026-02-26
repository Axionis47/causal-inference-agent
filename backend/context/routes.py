"""Thin authenticated HTTP boundary for the context capability."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import Response

from backend.auth import Db, Principal
from backend.config import settings
from backend.context import service
from backend.context.contracts import (
    PreparationView,
    PrepareInput,
    RetrievedContext,
    RevisionView,
    SaveInput,
    SourceText,
    SourceView,
    WorkspaceView,
)
from backend.models import DatasetVersion
from backend.storage import LocalFiles, files

router = APIRouter(prefix="/api/versions/{version_id}/context", tags=["context"])
Storage = Annotated[LocalFiles, Depends(files)]


def owned(version_id: UUID, db: Db, user: Principal) -> DatasetVersion:
    return service.owned_version(db, user.id, version_id)


Version = Annotated[DatasetVersion, Depends(owned)]


@router.get("", response_model=WorkspaceView)
def workspace(version: Version, db: Db) -> WorkspaceView:
    return service.workspace(db, version)


@router.post("/sources", response_model=SourceView, status_code=201)
def upload(
    version: Version,
    db: Db,
    storage: Storage,
    submission_id: Annotated[UUID, Form()],
    title: Annotated[str, Form(min_length=1, max_length=200)],
    kind: Annotated[Literal["file", "note"], Form()],
    file: Annotated[UploadFile, File()],
) -> SourceView:
    try:
        return service.upload_source(
            db,
            storage,
            version.id,
            submission_id,
            title,
            file.filename or "",
            kind,
            file.file.read(settings().context_source_bytes + 1),
        )
    finally:
        file.file.close()


@router.get("/sources/{source_id}", response_model=SourceText)
def read_source(source_id: UUID, version: Version, db: Db, storage: Storage) -> SourceText:
    return service.source_text(db, storage, version.id, source_id)


@router.get("/sources/{source_id}/file")
def download_source(source_id: UUID, version: Version, db: Db, storage: Storage) -> Response:
    from urllib.parse import quote

    source = service.source_record(db, version.id, source_id)
    return Response(
        storage.read(source.storage_key, source.sha256),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(source.filename, safe='')}"},
    )


@router.post("/preparations", response_model=PreparationView, status_code=202)
def prepare(
    body: PrepareInput, version: Version, db: Db, storage: Storage, user: Principal
) -> PreparationView:
    return service.prepare(db, storage, version, user.id, body)


@router.get("/preparations/{preparation_id}", response_model=PreparationView)
def preparation(preparation_id: UUID, version: Version, db: Db) -> PreparationView:
    return PreparationView.model_validate(service.preparation_record(db, version.id, preparation_id))


@router.post("/preparations/{preparation_id}/retry", response_model=PreparationView, status_code=202)
def retry(preparation_id: UUID, version: Version, db: Db) -> PreparationView:
    return service.retry_preparation(db, version.id, preparation_id)


@router.get("/revisions/{revision_id}", response_model=RevisionView)
def revision(revision_id: UUID, version: Version, db: Db) -> RevisionView:
    return service.revision_view(db, service.revision_record(db, version.id, revision_id))


@router.post("/revisions", response_model=RevisionView, status_code=201)
def save(body: SaveInput, version: Version, user: Principal, db: Db, storage: Storage) -> RevisionView:
    return service.save_revision(db, storage, version, user.id, body)


@router.get("/revisions/{revision_id}/read", response_model=RetrievedContext)
def read(
    revision_id: UUID,
    version: Version,
    db: Db,
    key: Annotated[list[str] | None, Query(max_length=100)] = None,
) -> RetrievedContext:
    return service.read_context(db, version.id, revision_id, key or [])
