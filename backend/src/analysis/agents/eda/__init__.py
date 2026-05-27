"""EDA agent package.

Exports only the typed output model. Agent class is imported directly
from .agent to avoid the base.state -> eda.agent -> base cycle.
"""

from .output import EDAResult

__all__ = ["EDAResult"]
