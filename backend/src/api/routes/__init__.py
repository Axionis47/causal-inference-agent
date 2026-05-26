"""API routes module."""

from src.download.api import router as download_router

from .health import router as health_router
from .jobs import router as jobs_router

__all__ = ["download_router", "health_router", "jobs_router"]
