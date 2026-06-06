"""Shared dispatch helpers used by every orchestrator implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.analysis.agents.base.state import JobStatus

if TYPE_CHECKING:
    import logging

    from src.analysis.agents.base.agent import BaseAgent
    from src.analysis.agents.base.state import AnalysisState


# The phase a specialist puts the job into when it starts running.
# Surfaced to the UI through state.status; used by both orchestrators
# so the user sees a consistent progress label regardless of mode.
AGENT_STATUS_MAP: dict[str, JobStatus] = {
    "domain_knowledge": JobStatus.PROFILING,
    "data_profiler": JobStatus.PROFILING,
    "eda_agent": JobStatus.EXPLORATORY_ANALYSIS,
    "causal_discovery": JobStatus.DISCOVERING_CAUSAL,
    "dag_expert": JobStatus.DISCOVERING_CAUSAL,
    "effect_estimator": JobStatus.ESTIMATING_EFFECTS,
    "sensitivity_analyst": JobStatus.SENSITIVITY_ANALYSIS,
    "notebook_generator": JobStatus.GENERATING_NOTEBOOK,
}


def validate_required_fields(
    specialists: dict[str, BaseAgent],
    agent_name: str,
    state: AnalysisState,
    logger: Any,
) -> list[str]:
    """Warn if a specialist's REQUIRED_STATE_FIELDS are not populated.

    Returns the list of missing fields. The caller decides whether to
    proceed; today both orchestrators warn-and-continue rather than
    refuse to dispatch.
    """
    agent = specialists.get(agent_name)
    if not agent:
        return []
    required = getattr(agent, "REQUIRED_STATE_FIELDS", [])
    missing = [f for f in required if getattr(state, f, None) is None]
    if missing:
        logger.warning(
            "required_fields_missing",
            agent=agent_name,
            missing_fields=missing,
        )
    return missing
