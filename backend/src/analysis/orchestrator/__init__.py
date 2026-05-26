"""Orchestrator agent module."""

from .base import Orchestrator
from .react import ReActOrchestrator
from .standard import StandardOrchestrator

# Transition alias. Existing scripts and notebooks may still import
# OrchestratorAgent; this keeps them working while internal callers
# migrate to StandardOrchestrator. Drop in a later cleanup commit.
OrchestratorAgent = StandardOrchestrator

__all__ = [
    "Orchestrator",
    "OrchestratorAgent",
    "ReActOrchestrator",
    "StandardOrchestrator",
]
