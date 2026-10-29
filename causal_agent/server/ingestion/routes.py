"""HTTP ingestion boundary, deliberately independent of desk sessions."""

import json
import logging
from functools import lru_cache
from typing import Annotated
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.responses import StreamingResponse

from causal_agent.common.config import ingestion
from causal_agent.server.ingestion import library
from causal_agent.server.ingestion.service import Conflict, Missing, Store
from causal_agent.server.ingestion.validation import InvalidUpload
from causal_agent.server.models import DatasetList, IngestedDataset

router = APIRouter()
log = logging.getLogger(__name__)


def principal() -> str:
    settings = ingestion()
    if settings.mode != "development":
        raise HTTPException(503, "Authentication is required. Development identity is disabled outside development.")
    return str(UUID(settings.user_id))


@lru_cache(maxsize=1)
def _store(settings):
    return Store(settings)


def store() -> Store:
    settings = ingestion()
    if not settings.database_url or not settings.access_key or not settings.secret_key:
        raise HTTPException(503, "Ingestion storage is not configured.")
    return _store(settings)


@router.post("/api/datasets", response_model=IngestedDataset, status_code=201)
def ingest_dataset(
    title: Annotated[str, Form(min_length=1, max_length=200)],
    idempotency_key: Annotated[str, Form(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")],
    file: Annotated[UploadFile, File()],
    owner: Annotated[str, Depends(principal)],
    storage: Annotated[Store, Depends(store)],
    context: Annotated[
        str, Form(max_length=600002, description="JSON-encoded original text, preserving line endings through browser multipart serialization")
    ] = '""',
):
    try:
        original_context = json.loads(context)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, "Context must be a JSON-encoded text string.") from exc
    if not isinstance(original_context, str) or len(original_context) > 100000 or "\x00" in original_context:
        raise HTTPException(422, "Context must be text of at most 100000 characters without NUL bytes.")
    if not title.strip():
        raise HTTPException(422, "Enter a dataset title.")
    filename = (file.filename or "").replace("\\", "/").split("/")[-1]
    if not filename or len(filename) > 255 or any(ord(c) < 32 for c in filename):
        raise HTTPException(422, "Choose a CSV with a valid filename (up to 255 characters).")
    try:
        return storage.ingest(owner, idempotency_key, title.strip(), original_context, filename, file.file)
    except InvalidUpload as exc:
        raise HTTPException(422, str(exc)) from exc
    except Conflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        log.exception("Ingestion storage failure")
        raise HTTPException(503, "Storage is temporarily unavailable. Retry this same submission.") from exc


@router.get("/api/datasets", response_model=DatasetList)
def list_datasets(
    response: Response,
    owner: Annotated[str, Depends(principal)],
    storage: Annotated[Store, Depends(store)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(max_length=1024)] = None,
):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return library.list_owned(storage, owner, limit, cursor)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        log.exception("Dataset listing failed")
        raise HTTPException(503, "Dataset storage is temporarily unavailable. Retry.") from exc


@router.get("/api/datasets/{dataset_id}", response_model=IngestedDataset)
def get_dataset(dataset_id: UUID, response: Response, owner: Annotated[str, Depends(principal)], storage: Annotated[Store, Depends(store)]):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return storage.get(owner, str(dataset_id))
    except Missing as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        log.exception("Dataset retrieval failed")
        raise HTTPException(503, "Dataset storage is temporarily unavailable. Retry.") from exc


@router.get("/api/datasets/{dataset_id}/file")
def download(dataset_id: UUID, owner: Annotated[str, Depends(principal)], storage: Annotated[Store, Depends(store)]):
    try:
        info, body = storage.download(owner, str(dataset_id))
    except Missing as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        log.exception("Dataset retrieval failed")
        raise HTTPException(503, "Dataset storage is temporarily unavailable. Retry.") from exc

    def chunks():
        try:
            yield from body.iter_chunks(64 * 1024)
        finally:
            body.close()

    return StreamingResponse(
        chunks(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(info['filename'], safe='')}",
            "Content-Length": str(info["size_bytes"]),
            "Cache-Control": "private, no-store",
        },
    )


class UploadLimit:
    """Bound the body before Starlette's multipart parser spools it to disk."""

    def __init__(self, app, max_bytes):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/api/datasets":
            return await self.app(scope, receive, send)
        total = 0

        async def bounded():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                raise HTTPException(413, "Upload exceeds the configured request size limit.")
            return message

        await self.app(scope, bounded, send)
