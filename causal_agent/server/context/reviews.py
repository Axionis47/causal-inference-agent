"""Review reads never prepare context or invoke a model."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response

from causal_agent.memory.context.review import RevisionConflict
from causal_agent.server.context.repository import Repository
from causal_agent.server.context.routes import HEADERS, Owner, Storage
from causal_agent.server.context.service import Prerequisite
from causal_agent.server.ingestion.service import Missing
from causal_agent.server.models import ReviewSave, ReviewStart, ReviewStatements, ReviewView

router = APIRouter()


def respond(fn):
    try:
        return fn()
    except Missing as exc:
        raise HTTPException(404, str(exc), headers=HEADERS) from exc
    except RevisionConflict as exc:
        raise HTTPException(
            409,
            {"code": "stale_revision", "message": str(exc), "current_revision_id": exc.current_id, "changed_statement_ids": exc.changed_ids},
            headers=HEADERS,
        ) from exc
    except Prerequisite as exc:
        raise HTTPException(409, {"code": "preparation_required", "message": str(exc)}, headers=HEADERS) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc), headers=HEADERS) from exc


@router.post("/api/datasets/{dataset_id}/reviews", response_model=ReviewView, status_code=201)
def start(dataset_id: UUID, body: ReviewStart, owner: Owner, storage: Storage, response: Response):
    response.headers.update(HEADERS)
    return respond(lambda: Repository(storage, owner, str(dataset_id)).start_review(body.question))


@router.get("/api/datasets/{dataset_id}/reviews/{review_id}", response_model=ReviewView)
def read(dataset_id: UUID, review_id: UUID, owner: Owner, storage: Storage, response: Response):
    response.headers.update(HEADERS)
    return respond(lambda: Repository(storage, owner, str(dataset_id)).review(str(review_id)))


@router.put("/api/datasets/{dataset_id}/reviews/{review_id}", response_model=ReviewView)
def save(dataset_id: UUID, review_id: UUID, body: ReviewSave, owner: Owner, storage: Storage, response: Response):
    response.headers.update(HEADERS)
    return respond(lambda: Repository(storage, owner, str(dataset_id)).save_review(str(review_id), body.generation, body.edits, body.dispositions))


@router.get("/api/datasets/{dataset_id}/reviews/{review_id}/statements", response_model=ReviewStatements)
def statements(
    dataset_id: UUID,
    review_id: UUID,
    owner: Owner,
    storage: Storage,
    response: Response,
    column_id: UUID | None = None,
    section: str = Query(default="dataset", pattern="^(dataset|relationship|concept)$"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    response.headers.update(HEADERS)

    def read_statements():
        repo = Repository(storage, owner, str(dataset_id))
        knowledge = repo.review_knowledge(str(review_id))
        items = knowledge.for_column(str(column_id)) if column_id else [s for s in knowledge.statements if s.scope == section and s.status != "withdrawn"]
        return ReviewStatements(statements=items[offset : offset + limit], next_offset=offset + limit if len(items) > offset + limit else None)

    return respond(read_statements)
