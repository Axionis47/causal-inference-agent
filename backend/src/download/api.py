"""FastAPI routes for the download subsystem.

Routes (under /api/v1/download):
    POST   /profile/kaggle           set/update key for a profile
    GET    /profile/kaggle           profile status (never returns the key)
    DELETE /profile/kaggle           clear the key
    POST   /downloads                start a download
    GET    /downloads                list records
    GET    /downloads/{id}           record by id
    GET    /downloads/{id}/events    SSE event stream
    DELETE /downloads/{id}           delete record and stored files

The actual work lives in pull, profile_store, download_store. This file
is only the HTTP boundary plus typed dependencies.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import get_settings
from src.domain.download import (
    DownloadRecord,
    DownloadRequest,
    KaggleProfile,
)
from src.download import (
    auth,
    download_store,
    events,
    profile_store,
    pull,
    transport,
)

router = APIRouter(prefix="/api/v1/download", tags=["download"])

DEFAULT_PROFILE_ID = "default"


def get_root() -> Path:
    return Path(get_settings().local_storage_path)


def get_encryption_key() -> str:
    key = get_settings().profile_encryption_key_value
    if not key:
        raise HTTPException(
            status_code=500,
            detail="PROFILE_ENCRYPTION_KEY is not configured on the server.",
        )
    return key


def get_event_bus() -> events.DownloadEventBus:
    return events.get_bus()


class SetKaggleKeyBody(BaseModel):
    username: str = Field(min_length=1)
    key: str = Field(min_length=1)
    validate_key: bool = True
    profile_id: str = DEFAULT_PROFILE_ID


class StartDownloadBody(BaseModel):
    kaggle_url: str = Field(min_length=1)
    profile_id: str = DEFAULT_PROFILE_ID


@router.post("/profile/kaggle", response_model=KaggleProfile)
async def set_kaggle_key(
    body: SetKaggleKeyBody,
    root: Path = Depends(get_root),
    encryption_key: str = Depends(get_encryption_key),
) -> KaggleProfile:
    if body.validate_key:
        result = await auth.validate_kaggle_key(body.username, body.key)
        if not result.valid:
            raise HTTPException(status_code=400, detail=result.error)

    profile = profile_store.set_kaggle_key(
        body.profile_id,
        username=body.username,
        key=body.key,
        root=root,
        encryption_key=encryption_key,
    )
    if body.validate_key:
        profile_store.mark_validated(body.profile_id, root=root)
        profile = profile_store.get_profile(body.profile_id, root=root) or profile
    return profile


@router.get("/profile/kaggle", response_model=KaggleProfile | None)
async def get_kaggle_profile(
    profile_id: str = DEFAULT_PROFILE_ID,
    root: Path = Depends(get_root),
) -> KaggleProfile | None:
    return profile_store.get_profile(profile_id, root=root)


@router.delete("/profile/kaggle", status_code=204)
async def delete_kaggle_profile(
    profile_id: str = DEFAULT_PROFILE_ID,
    root: Path = Depends(get_root),
) -> None:
    profile_store.clear_kaggle_key(profile_id, root=root)


@router.post("/downloads", response_model=DownloadRecord, status_code=202)
async def post_download(
    body: StartDownloadBody,
    root: Path = Depends(get_root),
    encryption_key: str = Depends(get_encryption_key),
    bus: events.DownloadEventBus = Depends(get_event_bus),
) -> DownloadRecord:
    req = DownloadRequest(profile_id=body.profile_id, kaggle_url=body.kaggle_url)
    download_id = await pull.start_download(
        req, root=root, encryption_key=encryption_key, bus=bus
    )
    record = download_store.load(download_id, root=root)
    if record is None:
        raise HTTPException(500, "Record disappeared after creation")
    return record


@router.get("/downloads", response_model=list[DownloadRecord])
async def list_downloads(
    profile_id: str | None = Query(default=None),
    root: Path = Depends(get_root),
) -> list[DownloadRecord]:
    return download_store.list_records(root=root, profile_id=profile_id)


@router.get("/downloads/{download_id}", response_model=DownloadRecord)
async def get_download(
    download_id: str,
    root: Path = Depends(get_root),
) -> DownloadRecord:
    record = download_store.load(download_id, root=root)
    if record is None:
        raise HTTPException(404, f"Download {download_id} not found")
    return record


@router.get("/downloads/{download_id}/events")
async def stream_download_events(
    download_id: str,
    root: Path = Depends(get_root),
    bus: events.DownloadEventBus = Depends(get_event_bus),
):
    record = download_store.load(download_id, root=root)
    if record is None:
        raise HTTPException(404, f"Download {download_id} not found")
    return transport.stream_download_events(bus, download_id)


@router.delete("/downloads/{download_id}", status_code=204)
async def delete_download(
    download_id: str,
    root: Path = Depends(get_root),
) -> None:
    record = download_store.load(download_id, root=root)
    if record is None:
        return
    if record.storage_path:
        shutil.rmtree(record.storage_path, ignore_errors=True)
    download_store.delete(download_id, root=root)
