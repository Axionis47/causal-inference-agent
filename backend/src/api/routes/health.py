"""Health check API routes."""

from fastapi import APIRouter

from src.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check endpoint for Kubernetes."""
    return {"status": "ready"}
