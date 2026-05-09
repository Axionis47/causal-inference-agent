"""Specialist roster for the react orchestrator.

The react orchestrator uses the ReAct-style effect estimator in place
of the standard one. Both are registered via the agent registry; this
roster picks which to expose under the "effect_estimator" key the
orchestrator actually dispatches to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents.registry import create_all_agents

if TYPE_CHECKING:
    from src.agents.base.agent import BaseAgent


def build_specialists() -> dict[str, BaseAgent]:
    """Return the specialist agents the react orchestrator dispatches to."""
    agents = create_all_agents()
    react_estimator = agents.pop("effect_estimator_react", None)
    if react_estimator is not None:
        agents["effect_estimator"] = react_estimator
    return agents
