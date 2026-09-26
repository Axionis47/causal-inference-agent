"""The app: JSON routes under /api, the built page for everything else."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from causal_agent.server import datasets as DS
from causal_agent.server import runs as R
from causal_agent.server.models import DatasetCreate, DatasetList, DatasetSummary, MessageIn, ProfileOut, RunFiles, SessionBrief, SessionView
from causal_agent.server.sessions import SessionBusy, SessionEnded, SessionManager, SessionState
from causal_agent.server.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings()
    app = FastAPI(title="Causal desk", docs_url="/api/docs", openapi_url="/api/openapi.json")
    mgr = SessionManager(s)
    app.state.settings, app.state.sessions = s, mgr

    # ------------------------------------------------------------------ datasets

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/datasets", response_model=DatasetList)
    def list_datasets() -> DatasetList:
        out = []
        for d in DS.list_datasets(s):
            if not d.shipped:
                try:
                    v = mgr.view(d.name)
                    d.session = SessionBrief(stage=v.stage, phase=v.phase, runs=len(v.runs))
                except Exception:
                    d.session = None
            out.append(d)
        return DatasetList(datasets=out)

    @app.post("/api/profile", response_model=ProfileOut, status_code=201)
    async def profile_upload(file: UploadFile) -> ProfileOut:
        content = await file.read()
        try:
            return DS.stage_upload(s, file.filename or "", content)
        except DS.BadUpload as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/datasets", response_model=DatasetSummary, status_code=201)
    def create_dataset(req: DatasetCreate) -> DatasetSummary:
        try:
            summary, _ = DS.create_dataset(s, req)
        except DS.Conflict as e:
            raise HTTPException(409, str(e)) from e
        except DS.NotFound as e:
            raise HTTPException(404, str(e)) from e
        sess = mgr.start(req.name)
        summary.session = SessionBrief(stage="busy" if sess.busy else "waiting", phase="before", runs=0)
        return summary

    @app.delete("/api/datasets/{name}", status_code=204)
    def delete_dataset(name: str) -> Response:
        run_dirs: list[str] = []
        if DS.read_meta(s, name):
            try:
                run_dirs = mgr.delete(name)
            except SessionBusy:
                raise HTTPException(409, "the conversation is busy; wait for it to finish, then delete") from None
        try:
            DS.delete_dataset(s, name, run_dirs)
        except DS.NotFound as e:
            raise HTTPException(404, str(e)) from e
        return Response(status_code=204)

    # ------------------------------------------------------------------ sessions

    def _view(name: str) -> SessionView:
        try:
            return mgr.view(name)
        except DS.NotFound as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/api/sessions/{name}", response_model=SessionView)
    def get_session(name: str) -> SessionView:
        return _view(name)

    @app.post("/api/sessions/{name}/messages", response_model=SessionView, status_code=202)
    def post_message(name: str, msg: MessageIn) -> SessionView:
        try:
            mgr.send(name, msg.text.strip())
        except DS.NotFound as e:
            raise HTTPException(404, str(e)) from e
        except SessionBusy:
            raise HTTPException(409, "still working on the last message") from None
        except SessionEnded:
            raise HTTPException(409, "the conversation has ended; start a new one") from None
        except SessionState as e:
            raise HTTPException(409, str(e)) from e
        return _view(name)

    @app.post("/api/sessions/{name}/resume", response_model=SessionView, status_code=202)
    def resume_session(name: str) -> SessionView:
        try:
            mgr.resume(name)
        except DS.NotFound as e:
            raise HTTPException(404, str(e)) from e
        except SessionBusy:
            raise HTTPException(409, "already running") from None
        except SessionEnded:
            raise HTTPException(409, "the conversation has ended; start a new one") from None
        return _view(name)

    @app.post("/api/sessions/{name}/analyses", response_model=SessionView, status_code=202)
    def new_analysis(name: str) -> SessionView:
        """A new question on the same file: a new thread over the memory as it stands; every run so far stays listed."""
        try:
            mgr.new_analysis(name)
        except DS.NotFound as e:
            raise HTTPException(404, str(e)) from e
        except SessionBusy:
            raise HTTPException(409, "still working; wait for it to finish") from None
        return _view(name)

    @app.post("/api/sessions/{name}/restart", response_model=SessionView, status_code=202, include_in_schema=False)
    def restart_session(name: str) -> SessionView:
        return new_analysis(name)

    # ------------------------------------------------------------------ runs

    @app.get("/api/runs/{run_id}", response_model=RunFiles)
    def run_files(run_id: str) -> RunFiles:
        try:
            return R.list_files(s, run_id)
        except R.NotFound:
            raise HTTPException(404, "no such run") from None

    @app.get("/api/runs/{run_id}/files/{filename}")
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
