"""ReAct orchestrator package.

Fully autonomous ReAct loop. No fixed workflow; the LLM observes the
state, reasons about what to do, dispatches specialists, and iterates
until the analysis is complete.
"""

from .agent import ReActOrchestrator

__all__ = ["ReActOrchestrator"]
