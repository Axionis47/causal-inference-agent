"""HTTP composition root. No domain state or validation belongs here."""

from fastapi import APIRouter

from backend.analysis.routes import router as analysis_router
from backend.chat.routes import router as chat_router
from backend.integration.readers import Analyses, Conversations


def routers() -> tuple[APIRouter, APIRouter]:
    return chat_router(Analyses()), analysis_router(Conversations())
