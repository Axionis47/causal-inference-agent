"""Propensity Score Diagnostics agent package.

Exports the typed output model. Agent class is imported directly from
.agent to avoid the base.state -> ps_diagnostics.agent -> base cycle.
"""

from .output import PSDiagnostics

__all__ = ["PSDiagnostics"]
