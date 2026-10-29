"""All chat reads and writes resolve an authenticated owner before touching runtime files."""

from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy import select

from causal_agent.memory.artifacts import Artifact
from causal_agent.server import artifacts as AV
from causal_agent.server.context.reviews import respond
from causal_agent.server.conversations import schema as S
from causal_agent.server.conversations import service
from causal_agent.server.conversations.commands import Repository as Commands
from causal_agent.server.conversations.publication import Repository
from causal_agent.server.ingestion.routes import principal, store
from causal_agent.server.ingestion.service import Missing, Store
from causal_agent.server.models import (
    AnalysisSummary,
    ArtifactSummary,
    ChatContextSummary,
    ChatStatements,
    ContextChanges,
    ConversationStart,
    ConversationView,
    EvidenceView,
    MessagePage,
    RunView,
)

router = APIRouter()
Owner = Annotated[str, Depends(principal)]
Storage = Annotated[Store, Depends(store)]


def bind(request: Request, name: UUID, owner: Owner, storage: Storage):
    try:
        chat = service.get(storage, owner, str(name))
        request.app.state.sessions.bind(str(name), storage, owner)
        return chat
    except Missing as exc:
        raise HTTPException(404, str(exc)) from exc


def authorize_run(run_id: str, request: Request, owner: Owner, storage: Storage):
    """Existing causal run files must be linked by an owned conversation record."""
    with storage.engine.connect() as db:
        results: Any = db.scalars(select(S.commands.c.result).where(S.commands.c.owner_id == owner, S.commands.c.status == "completed"))
        for result in results:
            directory = ((result or {}).get("record") or {}).get("run_dir")
            if directory and Path(directory).name == run_id:
                return
    raise HTTPException(404, "Run not found.")


@router.get("/api/conversations/{name}/messages", response_model=MessagePage)
def messages(
    name: UUID, owner: Owner, storage: Storage, chat: Annotated[dict, Depends(bind)], before: UUID | None = None, limit: int = Query(50, ge=1, le=100)
):
    def read():
        rows = Commands(storage, owner, str(name)).messages(str(before) if before else None, limit)
        return {"items": rows, "before": rows[-1]["id"] if len(rows) == limit else None}

    return respond(read)


@router.get("/api/conversations/{name}/context", response_model=ChatContextSummary)
def context_summary(name: UUID, owner: Owner, storage: Storage, chat: Annotated[dict, Depends(bind)]):
    repo = Commands(storage, owner, str(name))
    rev = repo.context.read(repo.state()["revision_id"])
    with storage.engine.connect() as db:
        latest = repo.context.head(db)["id"]
    return dict(revision_id=rev.id, revision=rev.number, columns=rev.knowledge.columns, dispositions=rev.knowledge.dispositions, latest_revision_id=latest)


@router.get("/api/conversations/{name}/context/changes", response_model=ContextChanges)
def context_changes(
    name: UUID, owner: Owner, storage: Storage, chat: Annotated[dict, Depends(bind)], offset: int = Query(0, ge=0), limit: int = Query(30, ge=1, le=100)
):
    repo = Commands(storage, owner, str(name))
    with storage.engine.connect() as db:
        before = repo.context.decode(repo.context.row(db, repo.state(db)["revision_id"]))
        after = repo.context.decode(repo.context.head(db))
    old = {s.id: s for s in before.knowledge.statements}
    new = {s.id: s for s in after.knowledge.statements}
    changes = [dict(before=old.get(key), after=new.get(key)) for key in sorted(old.keys() | new.keys()) if old.get(key) != new.get(key)]
    return dict(
        from_revision_id=before.id,
        to_revision_id=after.id,
        items=changes[offset : offset + limit],
        next_offset=offset + limit if len(changes) > offset + limit else None,
    )


@router.get("/api/conversations/{name}/context/statements", response_model=ChatStatements)
def context_statements(
    name: UUID,
    owner: Owner,
    storage: Storage,
    chat: Annotated[dict, Depends(bind)],
    column_id: UUID | None = None,
    section: str = Query("dataset", pattern="^(dataset|relationship|requirement)$"),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    def read():
        repo = Commands(storage, owner, str(name))
        rev = repo.context.read(repo.state()["revision_id"])
        items = (
            rev.knowledge.for_column(str(column_id)) if column_id else [s for s in rev.knowledge.statements if s.scope == section and s.status != "withdrawn"]
        )
        return {"revision_id": rev.id, "statements": items[offset : offset + limit], "next_offset": offset + limit if len(items) > offset + limit else None}

    return respond(read)


@router.get("/api/conversations/{name}/analyses", response_model=list[AnalysisSummary])
def analyses(
    name: UUID,
    request: Request,
    owner: Owner,
    storage: Storage,
    chat: Annotated[dict, Depends(bind)],
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
):
    with storage.engine.connect() as db:
        rows = db.execute(
            select(S.commands.c.id, S.commands.c.result)
            .where(
                S.commands.c.owner_id == owner,
                S.commands.c.conversation_id == str(name),
                S.commands.c.body["kind"].as_string() == "authorize",
                S.commands.c.status == "completed",
            )
            .order_by(S.commands.c.created_at.desc())
            .limit(limit)
            .offset(offset)
        ).mappings()
        return [
            dict(id=r["id"], **{k: r["result"]["record"].get(k) for k in ("index", "question", "family", "status", "lead_design_id")})
            for r in rows
            if (r["result"] or {}).get("record")
        ]


@router.get("/api/conversations/{name}/analyses/{analysis_id}", response_model=RunView)
def analysis(name: UUID, analysis_id: UUID, request: Request, owner: Owner, storage: Storage, chat: Annotated[dict, Depends(bind)]):
    from causal_agent.common.contracts import RunRecord
    from causal_agent.server.views import run_view

    with storage.engine.connect() as db:
        result = db.scalar(
            select(S.commands.c.result).where(
                S.commands.c.id == str(analysis_id),
                S.commands.c.owner_id == owner,
                S.commands.c.conversation_id == str(name),
                S.commands.c.status == "completed",
            )
        )
        if not result or not result.get("record"):
            raise HTTPException(404, "Analysis not found.")
        return run_view(RunRecord.model_validate(result["record"]), request.app.state.settings)


@router.post("/api/datasets/{dataset_id}/conversations", response_model=ConversationView, status_code=201)
def create(dataset_id: UUID, body: ConversationStart, request: Request, owner: Owner, storage: Storage):
    return respond(lambda: service.create(storage, owner, str(dataset_id), request.app.state.sessions, body.review_id, body.generation))


@router.get("/api/datasets/{dataset_id}/conversations", response_model=list[ConversationView])
def listing(dataset_id: UUID, owner: Owner, storage: Storage):
    try:
        return service.list_owned(storage, owner, str(dataset_id))
    except Missing as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/api/conversations/{name}/artifacts", response_model=list[ArtifactSummary])
def artifacts(
    name: UUID,
    owner: Owner,
    storage: Storage,
    chat: Annotated[dict, Depends(bind)],
    earlier: bool = False,
    offset: int = Query(0, ge=0),
    limit: int = Query(30, ge=1, le=50),
):
    repo = Repository(storage, owner, str(name))
    return [AV.view(str(name), a) for a in (repo.earlier(limit, offset) if earlier else repo.list(str(name), limit, offset))]


@router.get("/api/conversations/{name}/artifacts/{artifact_id}", response_model=Artifact)
def artifact(name: UUID, artifact_id: str, owner: Owner, storage: Storage, chat: Annotated[dict, Depends(bind)]):
    try:
        return Repository(storage, owner, str(name)).load(str(name), artifact_id).model_dump(mode="json")
    except Missing as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/api/conversations/{name}/artifacts/{artifact_id}/files/{file}")
def artifact_file(name: UUID, artifact_id: str, file: str, owner: Owner, storage: Storage, chat: Annotated[dict, Depends(bind)]):
    try:
        data = Repository(storage, owner, str(name)).read_file(artifact_id, file)
        return Response(
            data, media_type=AV.MEDIA.get("." + file.rsplit(".", 1)[-1], "application/octet-stream"), headers={"Cache-Control": "private, no-store"}
        )
    except Missing as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/api/conversations/{name}/evidence", response_model=EvidenceView)
def evidence(name: UUID, owner: Owner, storage: Storage, chat: Annotated[dict, Depends(bind)], address: str = Query(max_length=300)):
    repo = Commands(storage, owner, str(name))
    revision = repo.context.read(repo.state()["revision_id"])
    if address.startswith("context:"):
        statement = next((s for s in revision.knowledge.statements if s.id == address.removeprefix("context:")), None)
        if statement:
            return EvidenceView(
                title=statement.field.replace("_", " "),
                text=statement.value + " " + statement.qualifications,
                quotes=[c.quote for c in statement.citations] or [statement.source.text],
            )
    elif address.startswith("artifact:"):
        product = respond(lambda: Repository(storage, owner, str(name)).load(str(name), address.removeprefix("artifact:").split(".")[0]))
        return EvidenceView(
            title=product.summary,
            text=product.method + "\n" + "\n".join(product.findings + product.limitations),
            quotes=[f"{k.replace('_', ' ')}: {v}" for k, v in product.facts.items()],
        )
    elif address == "chat.question":
        return EvidenceView(title="Question", text=repo.state()["question"])
    elif address.startswith("operation:"):
        with storage.engine.connect() as db:
            operation = respond(lambda: repo.operation(db, address.removeprefix("operation:")))
        summary = (operation["result"] or {}).get("action_summary")
        if summary:
            return EvidenceView(title="Recorded action outcome", text=summary)
    raise HTTPException(404, "Evidence is not available in this chat.")
