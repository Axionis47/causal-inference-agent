import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.auth import router as auth_router
from backend.config import settings
from backend.context.routes import router as context_router
from backend.contracts import Error
from backend.datasets.routes import router as datasets_router
from backend.db import engine
from backend.integration.http import routers

app = FastAPI(
    title="Causal Workspace",
    version="0.1.0",
    responses={
        401: {"model": Error},
        403: {"model": Error},
        404: {"model": Error},
        409: {"model": Error},
        413: {"model": Error},
        422: {"model": Error},
        503: {"model": Error},
    },
)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
    return JSONResponse({"detail": "Check the submitted fields and try again."}, status_code=422)


@app.middleware("http")
async def boundaries(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        if request.headers.get("x-causal-client") != "web" or (
            origin and origin not in settings().allowed_origins.split(",")
        ):
            return JSONResponse({"detail": "This request is not allowed."}, status_code=403)
        length = request.headers.get("content-length")
        if not length or not length.isdigit():
            return JSONResponse({"detail": "Content-Length is required."}, status_code=411)
        if request.url.path == "/api/datasets":
            cap = settings().max_upload_bytes + 1024 * 1024
        elif request.url.path.startswith("/api/versions/") and request.url.path.endswith("/context/sources"):
            cap = settings().context_source_bytes + 64 * 1024
        elif request.url.path.startswith("/api/versions/") and request.url.path.endswith(
            "/context/revisions"
        ):
            cap = 2 * 1024 * 1024
        else:
            cap = 16_384
        if int(length) > cap:
            return JSONResponse(
                {"detail": "The request exceeds the allowed size for this operation."}, status_code=413
            )
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


@app.exception_handler(OSError)
@app.exception_handler(SQLAlchemyError)
async def unavailable(request: Request, error: Exception) -> JSONResponse:
    logging.getLogger(__name__).error("Persistence failure on %s", request.url.path, exc_info=error)
    return JSONResponse(
        {"detail": "Storage is unavailable. Please retry the same submission."}, status_code=503
    )


@app.exception_handler(ValueError)
async def invalid_value(request: Request, error: ValueError) -> JSONResponse:
    return JSONResponse({"detail": str(error)}, status_code=422)


@app.get("/api/health", tags=["system"])
def health() -> dict[str, str]:
    with engine().connect() as db:
        db.execute(text("SELECT 1"))
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(datasets_router)
app.include_router(context_router)
for router in routers():
    app.include_router(router)
