"""The app: JSON routes under /api, the built page for everything else."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from causal_agent.memory.context.conversation import ChatCommand
from causal_agent.server import runs as R
from causal_agent.server.context.reviews import respond
from causal_agent.server.conversations.manager import Manager
from causal_agent.server.conversations.routes import authorize_run, bind
from causal_agent.server.conversations.routes import router as conversation_router
from causal_agent.server.models import ChatOverview, RunFiles
from causal_agent.server.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings()
    mgr = Manager(s)

    @asynccontextmanager
    async def lifespan(app):
        yield
        mgr.pool.shutdown(wait=True)
        mgr.connection.close()

    app = FastAPI(title="Causal desk", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
    app.state.settings, app.state.sessions = s, mgr

    from causal_agent.common.config import ingestion
    from causal_agent.server.ingestion.routes import UploadLimit, router

    app.add_middleware(UploadLimit, max_bytes=ingestion().max_bytes + 1024 * 1024)
    app.include_router(router)
    from causal_agent.server.profiling.routes import router as profiling_router

    app.include_router(profiling_router)
    from causal_agent.server.context.routes import router as context_router

    app.include_router(context_router)
    from causal_agent.server.context.reviews import router as review_router

    app.include_router(review_router)
    app.include_router(conversation_router)

    # ------------------------------------------------------------------ datasets

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/ready")
    def ready() -> dict:
        import httpx

        from causal_agent.common import config
        from causal_agent.explore import sandbox

        box = config.get().sandbox
        try:
            sandbox.require_runtime()
            if box.agent_url:
                httpx.get(box.agent_url + "/.well-known/agent-card.json", timeout=2).raise_for_status()
        except (OSError, RuntimeError, httpx.HTTPError) as exc:
            raise HTTPException(503, "Exploration runtime unavailable: " + str(exc)) from exc
        return {"ok": True, "sandbox": "docker", "agent": "persistent" if box.agent_url else "on_demand"}

    # ------------------------------------------------------------------ sessions

    @app.get("/api/conversations/{name}", response_model=ChatOverview)
    def get_session(name: str, chat: Annotated[dict, Depends(bind)]):
        return respond(lambda: mgr.view(name))

    @app.post("/api/conversations/{name}/commands", response_model=ChatOverview, status_code=202)
    def post_command(name: str, body: ChatCommand, chat: Annotated[dict, Depends(bind)]):
        def submit():
            mgr.send(name, body)
            return mgr.view(name)

        return respond(submit)

    @app.post("/api/conversations/{name}/resume", response_model=ChatOverview, status_code=202)
    def resume_session(name: str, chat: Annotated[dict, Depends(bind)]):
        def resume():
            mgr.resume(name)
            return mgr.view(name)

        return respond(resume)

    # ------------------------------------------------------------------ runs

    @app.get("/api/runs/{run_id}", response_model=RunFiles, dependencies=[Depends(authorize_run)])
    def run_files(run_id: str) -> RunFiles:
        try:
            return R.list_files(s, run_id)
        except R.NotFound:
            raise HTTPException(404, "no such run") from None

    @app.get("/api/runs/{run_id}/files/{filename}", dependencies=[Depends(authorize_run)])
    def run_file(run_id: str, filename: str, raw: bool = False):
        try:
            path, media = R.file_path(s, run_id, filename)
        except R.NotFound:
            raise HTTPException(404, "no such file") from None
        if filename == "report.md":
            return PlainTextResponse(R.report_text(path, raw=raw))
        if media == "text/csv":
            return FileResponse(path, media_type=media, filename=filename)
        return FileResponse(path, media_type=media)

    # ------------------------------------------------------------------ the page

    dist = Path(s.dist)
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=str(dist / "assets")), name="assets")

    @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def page(path: str, request: Request):
        if path.startswith("api/"):
            raise HTTPException(404)
        index = dist / "index.html"
        if not index.exists():
            return PlainTextResponse("The page is not built. Run: npm run build --prefix web", status_code=503)
        candidate = (dist / path).resolve() if path else index
        if path and candidate.is_file() and dist.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index)

    return app
