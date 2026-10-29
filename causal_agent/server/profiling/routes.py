"""Owned profiling endpoints, independent of conversation and repair."""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response

from causal_agent.profile.report import ProfileView, SavedColumn
from causal_agent.server.ingestion.routes import principal, store
from causal_agent.server.ingestion.service import Conflict, Missing, Store
from causal_agent.server.profiling import service

router = APIRouter()
log = logging.getLogger(__name__)
Owner = Annotated[str, Depends(principal)]
Storage = Annotated[Store, Depends(store)]


def respond(response, fn, *args):
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return fn(*args)
    except Missing as exc:
        raise HTTPException(404, "Dataset not found.", headers={"Cache-Control": "private, no-store"}) from exc
    except Conflict as exc:
        raise HTTPException(409, "Profiling is already running. Reload or retry shortly.") from exc
    except Exception as exc:
        log.exception("Profile service failed")
        raise HTTPException(503, "Profiling is temporarily unavailable. Retry; any saved report is retained.") from exc


@router.get("/api/datasets/{dataset_id}/profile", response_model=ProfileView)
def get_profile(dataset_id: UUID, response: Response, owner: Owner, storage: Storage):
    return respond(response, service.view, storage, owner, str(dataset_id))


@router.post("/api/datasets/{dataset_id}/profile", response_model=ProfileView)
def generate_profile(dataset_id: UUID, response: Response, owner: Owner, storage: Storage):
    return respond(response, service.generate, storage, owner, str(dataset_id))


@router.get("/api/datasets/{dataset_id}/profiles/{profile_id}/columns/{column_id}", response_model=SavedColumn)
def get_column(dataset_id: UUID, profile_id: UUID, column_id: UUID, response: Response, owner: Owner, storage: Storage):
    return respond(response, service.column, storage, owner, str(dataset_id), str(profile_id), str(column_id))
