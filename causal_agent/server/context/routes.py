"""Owned context routes. Failure never replaces a previously published revision."""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response

from causal_agent.memory.context.extract import InvalidExtraction
from causal_agent.memory.context.views import ContextColumnView, ContextDatasetView, ContextView
from causal_agent.server.context import service
from causal_agent.server.ingestion.routes import principal, store
from causal_agent.server.ingestion.service import Conflict, Missing, Store

router = APIRouter()
log = logging.getLogger(__name__)
Owner = Annotated[str, Depends(principal)]
Storage = Annotated[Store, Depends(store)]
HEADERS = {"Cache-Control": "private, no-store"}


def respond(response, fn, *args):
    response.headers.update(HEADERS)
    try:
        return fn(*args)
    except Missing as exc:
        raise HTTPException(404, "Dataset not found.", headers=HEADERS) from exc
    except service.Prerequisite as exc:
        raise HTTPException(409, {"code": "profile_required", "message": str(exc)}, headers=HEADERS) from exc
    except Conflict as exc:
        raise HTTPException(409, {"code": "context_busy", "message": "Context is being prepared. Retry shortly."}, headers=HEADERS) from exc
    except InvalidExtraction as exc:
        raise HTTPException(502, "Context could not be validated. Retry; any saved context is retained.", headers=HEADERS) from exc
    except Exception as exc:
        log.exception("Context service failed")
        raise HTTPException(503, "Context is temporarily unavailable. Retry; any saved context is retained.", headers=HEADERS) from exc


@router.get("/api/datasets/{dataset_id}/context", response_model=ContextView)
def get_context(dataset_id: UUID, response: Response, owner: Owner, storage: Storage):
    return respond(response, service.view, storage, owner, str(dataset_id))


@router.post("/api/datasets/{dataset_id}/context", response_model=ContextView)
def prepare_context(dataset_id: UUID, response: Response, owner: Owner, storage: Storage):
    return respond(response, service.generate, storage, owner, str(dataset_id))


@router.get("/api/datasets/{dataset_id}/contexts/{revision_id}", response_model=ContextDatasetView)
def get_dataset_context(dataset_id: UUID, revision_id: UUID, response: Response, owner: Owner, storage: Storage):
    return respond(response, service.dataset, storage, owner, str(dataset_id), str(revision_id))


@router.get("/api/datasets/{dataset_id}/contexts/{revision_id}/columns/{column_id}", response_model=ContextColumnView)
def get_column_context(dataset_id: UUID, revision_id: UUID, column_id: UUID, response: Response, owner: Owner, storage: Storage):
    return respond(response, service.column, storage, owner, str(dataset_id), str(revision_id), str(column_id))
