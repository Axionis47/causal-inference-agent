"""Critique agent package.

Exports the typed output models. Agent class is imported directly from
.agent to avoid the base.state -> critique.agent -> base cycle.
"""

from .output import CritiqueDecision, CritiqueFeedback

__all__ = ["CritiqueDecision", "CritiqueFeedback"]
