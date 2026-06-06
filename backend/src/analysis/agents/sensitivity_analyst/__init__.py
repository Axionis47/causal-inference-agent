"""Sensitivity Analyst agent package.

Exports the typed output model. Agent class is imported directly from
.agent to avoid the base.state -> sensitivity_analyst.agent -> base
cycle.
"""

from .output import SensitivityResult

__all__ = ["SensitivityResult"]
