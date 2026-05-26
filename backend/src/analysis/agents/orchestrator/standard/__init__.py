"""Standard orchestrator package.

LLM-driven dispatch with a default workflow suggested in the system
prompt. The LLM can deviate from that order; the prompt is guidance,
not a hardcoded pipeline.
"""

from .agent import StandardOrchestrator
from .specialists import build_specialists

__all__ = ["StandardOrchestrator", "build_specialists"]
