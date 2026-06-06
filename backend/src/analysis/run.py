"""The analysis-side seam: run an analysis job from a JobInput.

This module is the single entry point analysis exposes to callers
outside the package. Today it is a stub so the seam location is in
the tree and any premature caller fails loudly with a clear pointer
to the plan stages that will wire it up.

The real implementation lands across two later stages:
- S6 relocates the orchestrator into ``analysis/orchestrator/`` and
  has it consume the substrate (AgentCtx, BudgetTracker, ToolCallLog,
  RetryPolicy, TraceBuffer, SSEEventBus, DecisionLog).
- S21 switches ``JobManager.create_job`` from instantiating an
  orchestrator directly to building a ``JobInput`` and calling this
  function. After that, this is how every analysis run starts.
"""
from __future__ import annotations

from src.domain.job import JobInput


async def run(job_input: JobInput) -> None:
    """Placeholder for the orchestrated analysis run.

    Raises NotImplementedError until S6 and S21 land. Callers that
    reach this today are running against an unmigrated layout and
    should keep going through ``JobManager`` directly.
    """
    raise NotImplementedError(
        "analysis.run() is not wired yet; see refactor plan S6 and S21"
    )
