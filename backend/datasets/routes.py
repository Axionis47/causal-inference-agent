from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response

from backend.auth import Db, Principal
from backend.config import settings
from backend.contracts import DatasetList, DatasetView, Preview
from backend.datasets import service
from backend.storage import LocalFiles, files

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
Storage = Annotated[LocalFiles, Depends(files)]
Page = Annotated[int, Query(ge=1, le=100_000)]
PageSize = Annotated[int, Query(ge=1, le=100)]


@router.get("", response_model=DatasetList)
def list_datasets(
    db: Db,
    user: Principal,
    page: Page = 1,
    page_size: PageSize = 20,
    search: Annotated[str, Query(max_length=200)] = "",
) -> DatasetList:
    return service.list_owned(db, user.id, page, page_size, search)


@router.post("", response_model=DatasetView, status_code=201)
def upload_dataset(
    db: Db,
    user: Principal,
    storage: Storage,
    title: Annotated[str, Form(min_length=1, max_length=200)],
    submission_id: Annotated[UUID, Form()],
    file: Annotated[UploadFile, File()],
) -> DatasetView:
    title = title.strip()
    filename = (file.filename or "").replace("\\", "/").split("/")[-1]
    if not title or not filename or len(filename) > 255 or any(ord(c) < 32 for c in filename):
        raise HTTPException(422, "Supply a title and a valid CSV filename.")
    try:
        content = file.file.read(settings().max_upload_bytes + 1)
        return service.upload(db, storage, user.id, submission_id, title, filename, content)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    finally:
        file.file.close()


@router.get("/{dataset_id}", response_model=DatasetView)
def get_dataset(dataset_id: UUID, db: Db, user: Principal) -> DatasetView:
    return service.view(*service.owned(db, user.id, dataset_id))


@router.get("/{dataset_id}/preview", response_model=Preview)
def get_preview(
    dataset_id: UUID, db: Db, user: Principal, storage: Storage, page: Page = 1, page_size: PageSize = 25
) -> Preview:
    _, version = service.owned(db, user.id, dataset_id)
    return service.preview(version, storage, page, page_size)


@router.get("/{dataset_id}/file")
def download(dataset_id: UUID, db: Db, user: Principal, storage: Storage) -> Response:
    from urllib.parse import quote

    _, version = service.owned(db, user.id, dataset_id)
    return Response(
        storage.read(version.storage_key, version.sha256),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(version.filename, safe='')}"},
    )
