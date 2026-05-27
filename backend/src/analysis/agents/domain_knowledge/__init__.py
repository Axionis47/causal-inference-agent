"""Domain knowledge agent: causal-inference context from dataset metadata.

`DomainKnowledgeAgent` lives at .agent and is intentionally not re-exported
here; importing it triggers a chain through `analysis.agents.base`, which
itself depends on this package's output types. Keep this module light.
"""

from .output import DomainKnowledge, Hypothesis, Uncertainty

__all__ = ["DomainKnowledge", "Hypothesis", "Uncertainty"]
