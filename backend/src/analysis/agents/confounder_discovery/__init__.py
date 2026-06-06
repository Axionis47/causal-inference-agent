"""Confounder Discovery agent package.

Exports only the typed output model. Agent class is imported directly
from .agent to avoid the base.state -> confounder_discovery.agent ->
base cycle.
"""

from .output import ConfounderAnalysis, ConfounderRecord

__all__ = ["ConfounderAnalysis", "ConfounderRecord"]
