"""Specialist roster for the standard orchestrator.

The standard orchestrator dispatches the registry's default specialists.
The ReAct-style effect estimator is registered alongside but not used
in standard mode, so it gets dropped from the roster here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.analysis.agents.registry import create_all_agents

if TYPE_CHECKING:
    from src.analysis.agents.base.agent import BaseAgent


def build_specialists() -> dict[str, BaseAgent]:
    """Return the specialist agents the standard orchestrator dispatches to."""
    agents = create_all_agents()
    agents.pop("effect_estimator_react", None)
    return agents
