"""Causal Discovery agent package.

Exports only the typed output models. Agent class is imported directly
from .agent to avoid the base.state -> causal_discovery.agent -> base
cycle. dag_expert reads CausalDAG from here too, since both agents
produce the same shape.
"""

from .output import CausalDAG, CausalEdge, CausalPair

__all__ = ["CausalDAG", "CausalEdge", "CausalPair"]
