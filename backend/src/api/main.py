"""FastAPI application entry point."""

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.rate_limit import limiter
from src.api.routes import health_router, jobs_router
from src.config import get_settings
from src.logging_config.structured import get_logger, setup_logging

# Set up logging
setup_logging()
logger = get_logger(__name__)


async def verify_api_key(x_api_key: str = Header(default=None)):
    """Verify API key if configured. Skip in dev mode when no key is set."""
    settings = get_settings()
    if settings.api_key_value is None:
        return  # No key configured = dev mode, skip auth
    if x_api_key != settings.api_key_value:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Agentic causal inference orchestration system",
        docs_url="/docs" if settings.enable_api_docs else None,
        redoc_url="/redoc" if settings.enable_api_docs else None,
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Accept"],
    )

    # Health routes are unauthenticated (needed for Cloud Run probes and load balancers)
    app.include_router(health_router)
    # Job routes require API key auth
    app.include_router(jobs_router, dependencies=[Depends(verify_api_key)])

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": "Agentic causal inference orchestration system",
            "docs": "/docs" if settings.enable_api_docs else None,
            "redoc": "/redoc" if settings.enable_api_docs else None,
            "endpoints": {
                "health": "/health",
                "jobs": "/jobs",
            },
        }

    @app.on_event("startup")
    async def startup_event():
        """Initialize resources on startup."""
        logger.info(
            "application_startup",
            app=settings.app_name,
            version=settings.app_version,
            environment=settings.environment,
        )

    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on shutdown."""
        logger.info("application_shutdown")

    return app


# Create the app instance
app = create_app()


def run_server():
    """Run the development server."""
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=not settings.is_production,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run_server()
