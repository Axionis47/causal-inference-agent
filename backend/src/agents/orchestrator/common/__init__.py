"""Shared infrastructure used by every orchestrator implementation.

The orchestrators (StandardOrchestrator and ReActOrchestrator) sit on top
of BaseAgent / ReActAgent, which already provide the shared tracing and
logging primitives:

    - AgentTrace recording inside AnalysisState.agent_traces
    - structlog event emission via self.logger
    - The execute_with_tracing wrapper that combines both at every
      agent boundary

This package adds the orchestrator-level helpers that are not part of
BaseAgent but are shared between orchestration modes:

    - AGENT_STATUS_MAP: which JobStatus to set when a given specialist
      starts, surfaced to the UI for progress tracking.
    - validate_required_fields: warn-only check that the specialist's
      REQUIRED_STATE_FIELDS are populated before dispatch.

Anything orchestrator-specific (the system prompt, the dispatch loop,
the tool definitions) lives in the orchestrator's own package, not here.
"""

from .context import summarize_dispatch_focus, summarize_progress
from .dispatch import AGENT_STATUS_MAP, validate_required_fields

__all__ = [
    "AGENT_STATUS_MAP",
    "summarize_dispatch_focus",
    "summarize_progress",
    "validate_required_fields",
]
